/*
 * Copyright (c) 2012, 2025, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 *
 */

#include "classfile/javaThreadStatus.hpp"
#include "code/codeCache.inline.hpp"
#include "code/debugInfoRec.hpp"
#include "code/nmethod.hpp"
#include "interpreter/interpreter.hpp"
#include "jfr/jfrEvents.hpp"
#include "jfr/periodic/sampling/jfrCPUTimeThreadSampler.hpp"
#include "jfr/periodic/sampling/jfrSampleMonitor.hpp"
#include "jfr/periodic/sampling/jfrSampleRequest.hpp"
#include "jfr/periodic/sampling/jfrThreadSampling.hpp"
#include "jfr/recorder/stacktrace/jfrStackTrace.hpp"
#include "jfr/utilities/jfrTypes.hpp"
#include "memory/resourceArea.hpp"
#include "oops/method.hpp"
#include "runtime/atomic.hpp"
#include "runtime/continuation.hpp"
#include "runtime/frame.inline.hpp"
#include "runtime/javaThread.inline.hpp"
#include "runtime/os.hpp"
#include "runtime/stackFrameStream.inline.hpp"
#include "utilities/globalDefinitions.hpp"
#include "utilities/ostream.hpp"
#include <cstdlib>
#include <cmath>
#include <cstring>

template <typename EventType>
static inline void send_sample_event(const JfrTicks& start_time, const JfrTicks& end_time, traceid sid, traceid tid) {
  EventType event(UNTIMED);
  event.set_starttime(start_time);
  event.set_endtime(end_time);
  event.set_sampledThread(tid);
  event.set_state(static_cast<u8>(JavaThreadStatus::RUNNABLE));
  event.set_stackTrace(sid);
  event.commit();
}

static inline void send_safepoint_latency_event(const JfrSampleRequest& request, const JfrTicks& end_time, traceid sid, const JavaThread* jt) {
  assert(jt != nullptr, "invariant");
  assert(!jt->jfr_thread_local()->has_cached_stack_trace(), "invariant");
  EventSafepointLatency event(UNTIMED);
  event.set_starttime(request._sample_ticks);
  event.set_endtime(end_time);
  if (event.should_commit()) {
    event.set_threadState(_thread_in_Java);
    jt->jfr_thread_local()->set_cached_stack_trace_id(sid);
    event.commit();
    jt->jfr_thread_local()->clear_cached_stack_trace();
  }
}

static inline bool is_interpreter(address pc) {
  return Interpreter::contains(pc);
}

static inline bool is_interpreter(const JfrSampleRequest& request) {
  return request._sample_bcp != nullptr;
}

static inline bool is_in_continuation(const frame& frame, JavaThread* jt) {
  return JfrThreadLocal::is_vthread(jt) &&
         (Continuation::is_frame_in_continuation(jt, frame) || Continuation::is_continuation_enterSpecial(frame));
}

// A sampled interpreter frame is handled differently from a sampled compiler frame.
//
// The JfrSampleRequest description partially describes a _potential_ interpreter Java frame.
// It's partial because the sampler thread only sets the fp and bcp fields.
//
// We want to ensure that what we discovered inside interpreter code _really_ is what we assume, a valid interpreter frame.
//
// Therefore, instead of letting the sampler thread read what it believes to be a Method*, we delay until we are at a safepoint to ensure the Method* is valid.
//
// If the JfrSampleRequest represents a valid interpreter frame, the Method* is retrieved and the sender frame is returned per the sender_frame.
//
// If it is not a valid interpreter frame, then the JfrSampleRequest is invalidated, and the current frame is returned per the sender_frame.
//
static bool compute_sender_frame(JfrSampleRequest& request, frame& sender_frame, bool& in_continuation, JavaThread* jt) {
  assert(is_interpreter(request), "invariant");
  assert(jt != nullptr, "invariant");
  assert(jt->has_last_Java_frame(), "invariant");

  // For a request representing an interpreter frame, request._sample_sp is actually the frame pointer, fp.
  const void* const sampled_fp = request._sample_sp;

  StackFrameStream stream(jt, false, false);

  // Search for the sampled interpreter frame and get its Method*.

  while (!stream.is_done()) {
    const frame* const frame = stream.current();
    assert(frame != nullptr, "invariant");
    const intptr_t* const real_fp = frame->real_fp();
    assert(real_fp != nullptr, "invariant");
    if (real_fp == sampled_fp && frame->is_interpreted_frame()) {
      Method* const method = frame->interpreter_frame_method();
      assert(method != nullptr, "invariant");
      request._sample_pc = method;
      // Got the Method*. Validate bcp.
      if (!method->is_native() &&  !method->contains(static_cast<address>(request._sample_bcp))) {
        request._sample_bcp = frame->interpreter_frame_bcp();
      }
      in_continuation = is_in_continuation(*frame, jt);
      break;
    }
    if (real_fp >= sampled_fp) {
      // What we sampled is not an official interpreter frame.
      // Invalidate the sample request and use current.
      request._sample_bcp = nullptr;
      sender_frame = *stream.current();
      in_continuation = is_in_continuation(sender_frame, jt);
      return true;
    }
    stream.next();
  }

  assert(!stream.is_done(), "invariant");

  // Step to sender.
  stream.next();

  // If the top frame is in a continuation, check that the sender frame is too.
  if (in_continuation && !is_in_continuation(*stream.current(), jt)) {
    // Leave sender frame empty.
    return true;
  }

  sender_frame = *stream.current();

  assert(request._sample_pc != nullptr, "invariant");
  assert(request._sample_bcp != nullptr, "invariant");
  assert(Method::is_valid_method(static_cast<const Method*>(request._sample_pc)), "invariant");
  assert(static_cast<const Method*>(request._sample_pc)->is_native() ||
         static_cast<const Method*>(request._sample_pc)->contains(static_cast<address>(request._sample_bcp)), "invariant");
  return true;
}

static inline const PcDesc* get_pc_desc(nmethod* nm, void* pc) {
  assert(nm != nullptr, "invariant");
  assert(pc != nullptr, "invariant");
  return nm->pc_desc_near(static_cast<address>(pc));
}

static inline bool is_valid(const PcDesc* pc_desc) {
  return pc_desc != nullptr && pc_desc->scope_decode_offset() != DebugInformationRecorder::serialized_null;
}

static bool compute_top_frame(const JfrSampleRequest& request, frame& top_frame, bool& in_continuation, JavaThread* jt, bool& biased) {
  assert(jt != nullptr, "invariant");

  if (!jt->has_last_Java_frame()) {
    return false;
  }

  if (is_interpreter(request)) {
    return compute_sender_frame(const_cast<JfrSampleRequest&>(request), top_frame, in_continuation, jt);
  }

  void* const sampled_pc = request._sample_pc;
  CodeBlob* sampled_cb;
  if (sampled_pc == nullptr || (sampled_cb = CodeCache::find_blob(sampled_pc)) == nullptr) {
    // A biased sample is requested or no code blob.
    top_frame = jt->last_frame();
    in_continuation = is_in_continuation(top_frame, jt);
    biased = true;
    return true;
  }

  // We will never describe a sample request that represents an unparsable stub or blob.
  assert(sampled_cb->frame_complete_offset() != CodeOffsets::frame_never_safe, "invariant");

  const void* const sampled_sp = request._sample_sp;
  assert(sampled_sp != nullptr, "invariant");

  nmethod* const sampled_nm = sampled_cb->as_nmethod_or_null();

  StackFrameStream stream(jt, false /* update registers */, false /* process frames */);

  if (stream.current()->is_safepoint_blob_frame()) {
    if (sampled_nm != nullptr) {
      // Move to the physical sender frame of the SafepointBlob stub frame using the frame size, not the logical iterator.
      const int safepoint_blob_stub_frame_size = stream.current()->cb()->frame_size();
      intptr_t* const sender_sp = stream.current()->unextended_sp() + safepoint_blob_stub_frame_size;
      if (sender_sp > sampled_sp) {
        const address saved_exception_pc = jt->saved_exception_pc();
        assert(saved_exception_pc != nullptr, "invariant");
        const nmethod* const exception_nm = CodeCache::find_blob(saved_exception_pc)->as_nmethod();
        assert(exception_nm != nullptr, "invariant");
        if (exception_nm == sampled_nm && sampled_nm->is_at_poll_return(saved_exception_pc)) {
          // We sit at the poll return site in the sampled compiled nmethod with only the return address on the stack.
          // The sampled_nm compiled frame is no longer extant, but we might be able to reconstruct a synthetic
          // compiled frame at this location. We do this by overlaying a reconstructed frame on top of
          // the huge SafepointBlob stub frame. Of course, the synthetic frame only contains random stack memory,
          // but it is safe because stack walking cares only about the form of the frame (i.e., an sp and a pc).
          // We also do not have to worry about stackbanging because we currently have a huge SafepointBlob stub frame
          // on the stack. For extra assurance, we know that we can create this frame size at this
          // very location because we just popped such a frame before we hit the return poll site.
          //
          // Let's attempt to correct for the safepoint bias.
          const PcDesc* const pc_desc = get_pc_desc(sampled_nm, sampled_pc);
          if (is_valid(pc_desc)) {
            intptr_t* const synthetic_sp = sender_sp - sampled_nm->frame_size();
            top_frame = frame(synthetic_sp, synthetic_sp, sender_sp, pc_desc->real_pc(sampled_nm), sampled_nm);
            in_continuation = is_in_continuation(top_frame, jt);
            return true;
          }
        }
      }
    }
    stream.next(); // skip the SafepointBlob stub frame
  }

  assert(!stream.current()->is_safepoint_blob_frame(), "invariant");

  biased = true;

  // Search the first frame that is above the sampled sp.
  for (; !stream.is_done(); stream.next()) {
    frame* const current = stream.current();

    if (current->real_fp() <= sampled_sp) {
      // Continue searching for a matching frame.
      continue;
    }

    if (sampled_nm == nullptr) {
      // The sample didn't have an nmethod; we decide to trace from its sender.
      // Another instance of safepoint bias.
      top_frame = *current;
      break;
    }

    // Check for a matching compiled method.
    if (current->cb()->as_nmethod_or_null() == sampled_nm) {
      if (current->pc() != sampled_pc) {
        // Let's adjust for the safepoint bias if we can.
        const PcDesc* const pc_desc = get_pc_desc(sampled_nm, sampled_pc);
        if (is_valid(pc_desc)) {
          current->adjust_pc(pc_desc->real_pc(sampled_nm));
          biased = false;
        }
      }
    }
    // Either a hit or a mismatched sample in which case we trace from the sender.
    // Yet another instance of safepoint bias,to be addressed with
    // more exact and stricter versions when parsable blobs become available.
    top_frame = *current;
    break;
  }

  in_continuation = is_in_continuation(top_frame, jt);
  return true;
}

static void record_thread_in_java(const JfrSampleRequest& request, const JfrTicks& now, const JfrThreadLocal* tl, JavaThread* jt, Thread* current) {
  assert(jt != nullptr, "invariant");
  assert(tl != nullptr, "invariant");
  assert(current != nullptr, "invariant");

  frame top_frame;
  bool biased = false;
  bool in_continuation;
  if (!compute_top_frame(request, top_frame, in_continuation, jt, biased)) {
    return;
  }

  traceid sid;
  {
    ResourceMark rm(current);
    JfrStackTrace stacktrace;
    if (!stacktrace.record(jt, top_frame, in_continuation, request)) {
      // Unable to record stacktrace. Fail.
      return;
    }
    sid = JfrStackTraceRepository::add(stacktrace);
  }
  assert(sid != 0, "invariant");
  const traceid tid = in_continuation ? tl->vthread_id_with_epoch_update(jt) : JfrThreadLocal::jvm_thread_id(jt);
  send_sample_event<EventExecutionSample>(request._sample_ticks, now, sid, tid);
  if (current == jt) {
    send_safepoint_latency_event(request, now, sid, jt);
  }
}

#ifdef LINUX
static void record_cpu_time_thread(const JfrCPUTimeSampleRequest& request, const JfrTicks& now, const JfrThreadLocal* tl, JavaThread* jt, Thread* current) {
  assert(jt != nullptr, "invariant");
  assert(tl != nullptr, "invariant");
  assert(current != nullptr, "invariant");
  frame top_frame;
  bool biased = false;
  bool in_continuation = false;
  bool could_compute_top_frame = compute_top_frame(request._request, top_frame, in_continuation, jt, biased);
  const traceid tid = in_continuation ? tl->vthread_id_with_epoch_update(jt) : JfrThreadLocal::jvm_thread_id(jt);

  if (!could_compute_top_frame) {
    JfrCPUTimeThreadSampling::send_empty_event(request._request._sample_ticks, tid, request._cpu_time_period);
    return;
  }
  traceid sid;
  {
    ResourceMark rm(current);
    JfrStackTrace stacktrace;
    if (!stacktrace.record(jt, top_frame, in_continuation, request._request)) {
      // Unable to record stacktrace. Fail.
      JfrCPUTimeThreadSampling::send_empty_event(request._request._sample_ticks, tid, request._cpu_time_period);
      return;
    }
    sid = JfrStackTraceRepository::add(stacktrace);
  }
  assert(sid != 0, "invariant");


  JfrCPUTimeThreadSampling::send_event(request._request._sample_ticks, sid, tid, request._cpu_time_period, biased);
  if (current == jt) {
    send_safepoint_latency_event(request._request, now, sid, jt);
  }
}
#endif

static void drain_enqueued_requests(const JfrTicks& now, JfrThreadLocal* tl, JavaThread* jt, Thread* current) {
  assert(tl != nullptr, "invariant");
  assert(jt != nullptr, "invariant");
  assert(current != nullptr, "invariant");
  assert(jt->jfr_thread_local() == tl, "invariant");
  assert_lock_strong(tl->sample_monitor());
  if (tl->has_enqueued_requests()) {
    for (const JfrSampleRequest& request : *tl->sample_requests()) {
      record_thread_in_java(request, now, tl, jt, current);
    }
    tl->clear_enqueued_requests();
  }
  assert(!tl->has_enqueued_requests(), "invariant");
}


struct DrainStats {
  static const int HISTOGRAM_BUCKETS = 1000;
  static const long MAX_DRAIN_TIME_NS = 1000000000L; // 1 second in nanoseconds
  static const long MAX_EVENT_COUNT = 5000;

  volatile long _drains;
  volatile long _drain_time_sum;
  volatile long _drain_time_max;
  volatile long _drain_time_min;
  volatile long _event_sum;
  volatile long _event_max;
  volatile long _event_min;
  volatile long _last_print_time;
  volatile long _start_time;  // Wall clock time when first operation occurred

  // Time histogram: [0] = underflow, [1-1000] = buckets, [1001] = overflow
  volatile long _drain_time_histogram[HISTOGRAM_BUCKETS + 2];
  // Event counts: [0-1000] = exact counts, [1001] = overflow (>1000)
  volatile long _event_histogram[MAX_EVENT_COUNT + 2];

  DrainStats() : _drains(0), _drain_time_sum(0), _drain_time_max(0), _drain_time_min(LONG_MAX),
                 _event_sum(0), _event_max(0), _event_min(LONG_MAX), _last_print_time(0), _start_time(0) {
    for (int i = 0; i < HISTOGRAM_BUCKETS + 2; i++) {
      _drain_time_histogram[i] = 0;
    }
    for (int i = 0; i < MAX_EVENT_COUNT + 2; i++) {
      _event_histogram[i] = 0;
    }
  }

  void update(long new_time, long new_events = 0) {
    // Initialize start time on first update
    if (_start_time == 0) {
      _start_time = os::javaTimeNanos();
    }

    Atomic::inc(&_drains);
    Atomic::add(&_drain_time_sum, new_time);

    // Update drain time max
    if (new_time > _drain_time_max) {
      while (true) {
        long old_max = _drain_time_max;
        if (new_time <= old_max || Atomic::cmpxchg(&_drain_time_max, old_max, new_time) == old_max) {
          break;
        }
      }
    }

    // Update drain time min
    if (new_time < _drain_time_min) {
      while (true) {
        long old_min = _drain_time_min;
        if (new_time >= old_min || Atomic::cmpxchg(&_drain_time_min, old_min, new_time) == old_min) {
          break;
        }
      }
    }

    // Update drain time histogram (logarithmic)
    int time_bucket;
    if (new_time <= 0) {
      time_bucket = 0; // underflow
    } else if (new_time >= MAX_DRAIN_TIME_NS) {
      time_bucket = HISTOGRAM_BUCKETS + 1; // overflow
    } else {
      // Logarithmic bucketing: log10(new_time + 1) mapped to [1, HISTOGRAM_BUCKETS]
      double log_time = log10((double)(new_time + 1));
      double log_max = log10((double)(MAX_DRAIN_TIME_NS + 1));
      time_bucket = 1 + (int)((log_time * HISTOGRAM_BUCKETS) / log_max);
      if (time_bucket > HISTOGRAM_BUCKETS) time_bucket = HISTOGRAM_BUCKETS;
    }
    Atomic::inc(&_drain_time_histogram[time_bucket]);

    if (new_events > 0) {
      Atomic::add(&_event_sum, new_events);

      // Update event max
      if (new_events > _event_max) {
        while (true) {
          long old_max = _event_max;
          if (new_events <= old_max || Atomic::cmpxchg(&_event_max, old_max, new_events) == old_max) {
            break;
          }
        }
      }

      // Update event min
      if (new_events < _event_min) {
        while (true) {
          long old_min = _event_min;
          if (new_events >= old_min || Atomic::cmpxchg(&_event_min, old_min, new_events) == old_min) {
            break;
          }
        }
      }

      // Update event histogram
      int event_bucket;
      if (new_events > MAX_EVENT_COUNT) {
        event_bucket = MAX_EVENT_COUNT + 1; // overflow bucket
      } else {
        event_bucket = (int)new_events; // direct mapping for 0-200
      }
      Atomic::inc(&_event_histogram[event_bucket]);
    }
  }

  void print(const char* name) {
    if (_drains == 0) {
      return;
    }

    long drains = Atomic::load(&_drains);
    long time_sum = Atomic::load(&_drain_time_sum);
    long time_max = Atomic::load(&_drain_time_max);
    long time_min = (_drain_time_min == LONG_MAX) ? 0 : Atomic::load(&_drain_time_min);
    long event_sum = Atomic::load(&_event_sum);
    long event_max = Atomic::load(&_event_max);
    long event_min = (_event_min == LONG_MAX) ? 0 : Atomic::load(&_event_min);

    // Calculate percentiles
    long time_median = calculate_time_percentile(50.0);
    long time_p90 = calculate_time_percentile(90.0);
    long time_p95 = calculate_time_percentile(95.0);
    long time_p99 = calculate_time_percentile(99.0);
    long time_p999 = calculate_time_percentile(99.9);

    long event_median = calculate_event_percentile(50.0);
    long event_p90 = calculate_event_percentile(90.0);
    long event_p95 = calculate_event_percentile(95.0);
    long event_p99 = calculate_event_percentile(99.0);
    long event_p999 = calculate_event_percentile(99.9);

    // Human readable summary
    long current_wall_time = os::javaTimeNanos();
    long wall_runtime_ns = (_start_time > 0) ? (current_wall_time - _start_time) : 0;
    double wall_runtime_seconds = wall_runtime_ns / 1000000000.0;

    printf("\n=== %s Drain Statistics ===\n", name);
    printf("Requests: %ld\n", drains);
    printf("Runtime: %.3f seconds (%.1f minutes)\n", wall_runtime_seconds, wall_runtime_seconds / 60.0);
    printf("Request Rate: %.2f requests/second\n", wall_runtime_seconds > 0 ? drains / wall_runtime_seconds : 0.0);
    printf("Time (ns): sum=%ld, avg=%ld, min=%ld, max=%ld, median=%ld, p95=%ld, p99=%ld, p99.9=%ld\n",
           time_sum, drains > 0 ? time_sum / drains : 0, time_min, time_max, time_median, time_p95, time_p99, time_p999);
    printf("Events: sum=%ld, avg=%.2f, min=%ld, max=%ld, median=%ld, p95=%ld, p99=%ld, p99.9=%ld\n",
           event_sum, drains > 0 ? event_sum * 1.0 / drains : 0.0, event_min, event_max, event_median, event_p95, event_p99, event_p999);

    // ASCII visualization of time histogram (logarithmic)
    printf("\nTime Distribution (Log Scale, 0-1s):\n");

    // Define bucket ranges and calculate their actual time ranges
    struct { int start, end; } bucket_ranges[] = {
      {0, 1},     // underflow
      {1, 51},    // buckets 1-50
      {51, 101},  // buckets 51-100
      {101, 201}, // buckets 101-200
      {201, 301}, // buckets 201-300
      {301, 401}, // buckets 301-400
      {401, 501}, // buckets 401-500
      {501, 601}, // buckets 501-600
      {601, 701}, // buckets 601-700
      {701, 801}, // buckets 701-800
      {801, 901}, // buckets 801-900
      {901, 1001}, // buckets 901-1000
      {1001, 1002} // overflow
    };

    long time_max_count = 0;
    for (int j = 0; j < 13; j++) {
      long total = 0;
      for (int k = bucket_ranges[j].start; k < bucket_ranges[j].end && k < HISTOGRAM_BUCKETS + 2; k++) {
        total += Atomic::load(&_drain_time_histogram[k]);
      }
      if (total > time_max_count) time_max_count = total;
    }

    if (time_max_count > 0) {
      for (int j = 0; j < 13; j++) {
        long total = 0;
        for (int k = bucket_ranges[j].start; k < bucket_ranges[j].end && k < HISTOGRAM_BUCKETS + 2; k++) {
          total += Atomic::load(&_drain_time_histogram[k]);
        }
        if (total == 0) continue;

        // Calculate the actual time range for this bucket range
        char label[50];
        if (bucket_ranges[j].start == 0) {
          strcpy(label, "≤0ns");
        } else if (bucket_ranges[j].start >= HISTOGRAM_BUCKETS + 1) {
          strcpy(label, "≥1s");
        } else {
          // Calculate time range using the same logarithmic formula as update()
          double log_max = log10((double)(MAX_DRAIN_TIME_NS + 1));

          // Start of range
          double log_min_start = (double)(bucket_ranges[j].start - 1) * log_max / HISTOGRAM_BUCKETS;
          long time_start = (long)(pow(10.0, log_min_start)) - 1;
          if (time_start < 0) time_start = 0;

          // End of range
          double log_min_end = (double)(bucket_ranges[j].end - 2) * log_max / HISTOGRAM_BUCKETS;
          long time_end = (long)(pow(10.0, log_min_end)) - 1;
          if (time_end >= MAX_DRAIN_TIME_NS) time_end = MAX_DRAIN_TIME_NS - 1;

          // Format the label with appropriate units
          if (time_end < 1000) {
            snprintf(label, sizeof(label), "%ldns-%ldns", time_start, time_end);
          } else if (time_end < 1000000) {
            snprintf(label, sizeof(label), "%.1fμs-%.1fμs", time_start/1000.0, time_end/1000.0);
          } else if (time_end < 1000000000) {
            snprintf(label, sizeof(label), "%.2fms-%.2fms", time_start/1000000.0, time_end/1000000.0);
          } else {
            snprintf(label, sizeof(label), "%.3fs-%.3fs", time_start/1000000000.0, time_end/1000000000.0);
          }
        }

        int bar_len = (int)((total * 40) / time_max_count);
        printf("%15s: ", label);
        for (int k = 0; k < bar_len; k++) printf("█");
        printf(" %ld\n", total);
      }
    }

    // ASCII visualization of event histogram
    printf("\nEvent Count Distribution:\n");
    long event_max_count = 0;
    for (int i = 0; i < MAX_EVENT_COUNT + 2; i++) {
      long count = Atomic::load(&_event_histogram[i]);
      if (count > event_max_count) event_max_count = count;
    }

    if (event_max_count > 0) {
      // Group events for display
      struct { int start, end; const char* label; } ranges[] = {
        {0, 1, "0"},
        {1, 6, "1-5"},
        {6, 11, "6-10"},
        {11, 21, "11-20"},
        {21, 51, "21-50"},
        {51, 101, "51-100"},
        {101, 201, "101-200"},
        {201, 501, "201-500"},
        {501, 1001, "501-1000"},
        {1001, 1002, ">1000"}
      };

      for (int j = 0; j < 10; j++) {
        long total = 0;
        for (int k = ranges[j].start; k < ranges[j].end && k < MAX_EVENT_COUNT + 2; k++) {
          total += Atomic::load(&_event_histogram[k]);
        }
        if (total == 0) continue;

        int bar_len = (int)((total * 30) / event_max_count);
        printf("%8s: ", ranges[j].label);
        for (int k = 0; k < bar_len; k++) printf("█");
        printf(" %ld\n", total);
      }
    }

    // Machine readable output with structured histogram data
    long json_current_time = os::javaTimeNanos();
    long json_runtime_ns = (_start_time > 0) ? (json_current_time - _start_time) : 0;
    double json_runtime_seconds = json_runtime_ns / 1000000000.0;

    // Build JSON string in memory to avoid printf buffering issues
    stringStream json_buffer;

    json_buffer.print("DRAIN_STATS_JSON:{\"name\":\"%s\",\"drains\":%ld,\"runtime_ns\":%ld,\"runtime_seconds\":%.3f,\"time\":{\"sum\":%ld,\"avg\":%ld,\"min\":%ld,\"max\":%ld,\"median\":%ld,\"p90\":%ld,\"p95\":%ld,\"p99\":%ld,\"p99_9\":%ld},\"events\":{\"sum\":%ld,\"avg\":%.2f,\"min\":%ld,\"max\":%ld,\"median\":%ld,\"p90\":%ld,\"p95\":%ld,\"p99\":%ld,\"p99_9\":%ld},\"time_histogram\":[",
           name, drains, json_runtime_ns, json_runtime_seconds, time_sum, drains > 0 ? time_sum / drains : 0, time_min, time_max, time_median, time_p90, time_p95, time_p99, time_p999,
           event_sum, drains > 0 ? event_sum * 1.0 / drains : 0.0, event_min, event_max, event_median, event_p90, event_p95, event_p99, event_p999);

    // Generate structured time histogram with from/to ranges
    bool first_time = true;
    for (int i = 0; i < HISTOGRAM_BUCKETS + 2; i++) {
      long count = Atomic::load(&_drain_time_histogram[i]);
      if (count > 0) {  // Only include non-zero buckets to reduce output size
        if (!first_time) json_buffer.print(",");
        first_time = false;

        if (i == 0) {
          json_buffer.print("{\"from\":0,\"to\":1,\"count\":%ld,\"range\":\"underflow\"}", count);
        } else if (i >= HISTOGRAM_BUCKETS + 1) {
          json_buffer.print("{\"from\":1000000000,\"to\":null,\"count\":%ld,\"range\":\"overflow\"}", count);
        } else {
          // Convert bucket back to time range (reverse of logarithmic bucketing)
          // Forward formula: time_bucket = 1 + (int)((log_time * HISTOGRAM_BUCKETS) / log_max)
          // Where log_time = log10(new_time + 1) and log_max = log10(MAX_DRAIN_TIME_NS + 1)
          double log_max = log10((double)(MAX_DRAIN_TIME_NS + 1));

          // Calculate time range for this bucket
          double log_min = (double)(i - 1) * log_max / HISTOGRAM_BUCKETS;
          double log_max_bucket = (double)i * log_max / HISTOGRAM_BUCKETS;

          long time_min_ns = (long)(pow(10.0, log_min)) - 1;
          long time_max_ns = (long)(pow(10.0, log_max_bucket)) - 1;

          // Ensure bounds are reasonable
          if (time_min_ns < 0) time_min_ns = 0;
          if (time_max_ns >= MAX_DRAIN_TIME_NS) time_max_ns = MAX_DRAIN_TIME_NS - 1;

          json_buffer.print("{\"from\":%ld,\"to\":%ld,\"count\":%ld}", time_min_ns, time_max_ns, count);
        }
      }
    }
    json_buffer.print("],\"event_histogram\":[");

    // Generate structured event histogram
    bool first_event = true;
    for (int i = 0; i < MAX_EVENT_COUNT + 2; i++) {
      long count = Atomic::load(&_event_histogram[i]);
      if (count > 0) {  // Only include non-zero buckets
        if (!first_event) json_buffer.print(",");
        first_event = false;

        if (i >= MAX_EVENT_COUNT + 1) {
          json_buffer.print("{\"from\":%ld,\"to\":null,\"count\":%ld,\"range\":\"overflow\"}", (long)MAX_EVENT_COUNT, count);
        } else {
          json_buffer.print("{\"from\":%ld,\"to\":%ld,\"count\":%ld}", (long)i, (long)i, count);
        }
      }
    }
    json_buffer.print("]}");

    // Output the complete JSON in one atomic operation
    tty->print_cr("%s", json_buffer.base());
  }

  long count() const {
    return Atomic::load(&_drains);
  }

  bool should_print() {
    long current_time = os::javaTimeNanos();
    long last_print = Atomic::load(&_last_print_time);
    const long THIRTY_SECONDS_NS = 30000000000L; // 30 seconds in nanoseconds

    if (current_time - last_print >= THIRTY_SECONDS_NS) {
      // Try to update the last print time atomically
      if (Atomic::cmpxchg(&_last_print_time, last_print, current_time) == last_print) {
        return true; // We successfully updated the time, so we should print
      }
    }
    return false; // Either not enough time has passed or another thread is printing
  }

  // Combine statistics from multiple DrainStats objects
  static void print_combined(const char* name, const DrainStats& stats1, const DrainStats& stats2,
                           const DrainStats& stats3, const DrainStats& stats4) {
    // Aggregate basic statistics
    long total_drains = stats1.count() + stats2.count() + stats3.count() + stats4.count();
    if (total_drains == 0) {
      return;
    }

    long total_time_sum = Atomic::load(&stats1._drain_time_sum) + Atomic::load(&stats2._drain_time_sum) +
                         Atomic::load(&stats3._drain_time_sum) + Atomic::load(&stats4._drain_time_sum);
    long total_event_sum = Atomic::load(&stats1._event_sum) + Atomic::load(&stats2._event_sum) +
                          Atomic::load(&stats3._event_sum) + Atomic::load(&stats4._event_sum);

    // Find combined min/max
    long combined_time_min = LONG_MAX;
    long combined_time_max = 0;
    long combined_event_min = LONG_MAX;
    long combined_event_max = 0;

    const DrainStats* all_stats[] = {&stats1, &stats2, &stats3, &stats4};
    for (int i = 0; i < 4; i++) {
      if (all_stats[i]->count() > 0) {
        long time_min = (all_stats[i]->_drain_time_min == LONG_MAX) ? 0 : Atomic::load(&all_stats[i]->_drain_time_min);
        long time_max = Atomic::load(&all_stats[i]->_drain_time_max);
        long event_min = (all_stats[i]->_event_min == LONG_MAX) ? 0 : Atomic::load(&all_stats[i]->_event_min);
        long event_max = Atomic::load(&all_stats[i]->_event_max);

        if (time_min < combined_time_min) combined_time_min = time_min;
        if (time_max > combined_time_max) combined_time_max = time_max;
        if (event_min < combined_event_min) combined_event_min = event_min;
        if (event_max > combined_event_max) combined_event_max = event_max;
      }
    }
    if (combined_time_min == LONG_MAX) combined_time_min = 0;
    if (combined_event_min == LONG_MAX) combined_event_min = 0;

    // Calculate combined percentiles
    long combined_time_median = calculate_combined_time_percentile(stats1, stats2, stats3, stats4, 50.0);
    long combined_time_p90 = calculate_combined_time_percentile(stats1, stats2, stats3, stats4, 90.0);
    long combined_time_p95 = calculate_combined_time_percentile(stats1, stats2, stats3, stats4, 95.0);
    long combined_time_p99 = calculate_combined_time_percentile(stats1, stats2, stats3, stats4, 99.0);
    long combined_time_p999 = calculate_combined_time_percentile(stats1, stats2, stats3, stats4, 99.9);

    long combined_event_median = calculate_combined_event_percentile(stats1, stats2, stats3, stats4, 50.0);
    long combined_event_p90 = calculate_combined_event_percentile(stats1, stats2, stats3, stats4, 90.0);
    long combined_event_p95 = calculate_combined_event_percentile(stats1, stats2, stats3, stats4, 95.0);
    long combined_event_p99 = calculate_combined_event_percentile(stats1, stats2, stats3, stats4, 99.0);
    long combined_event_p999 = calculate_combined_event_percentile(stats1, stats2, stats3, stats4, 99.9);

    // Human readable summary
    printf("\n=== %s Combined Drain Statistics ===\n", name);
    printf("Requests: %ld\n", total_drains);
    printf("Time (ns): sum=%ld, avg=%ld, min=%ld, max=%ld, median=%ld, p95=%ld, p99=%ld, p99.9=%ld\n",
           total_time_sum, total_drains > 0 ? total_time_sum / total_drains : 0, combined_time_min, combined_time_max,
           combined_time_median, combined_time_p95, combined_time_p99, combined_time_p999);
    printf("Events: sum=%ld, avg=%.2f, min=%ld, max=%ld, median=%ld, p95=%ld, p99=%ld, p99.9=%ld\n",
           total_event_sum, total_drains > 0 ? total_event_sum * 1.0 / total_drains : 0.0, combined_event_min, combined_event_max,
           combined_event_median, combined_event_p95, combined_event_p99, combined_event_p999);

    // Combine histograms
    long combined_time_histogram[HISTOGRAM_BUCKETS + 2] = {0};
    long combined_event_histogram[MAX_EVENT_COUNT + 2] = {0};

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < HISTOGRAM_BUCKETS + 2; j++) {
        combined_time_histogram[j] += Atomic::load(&all_stats[i]->_drain_time_histogram[j]);
      }
      for (int j = 0; j < MAX_EVENT_COUNT + 2; j++) {
        combined_event_histogram[j] += Atomic::load(&all_stats[i]->_event_histogram[j]);
      }
    }

    // ASCII visualization of combined time histogram
    printf("\nTime Distribution (Log Scale, 0-1s):\n");

    // Define bucket ranges and calculate their actual time ranges
    struct { int start, end; } bucket_ranges[] = {
      {0, 1},     // underflow
      {1, 51},    // buckets 1-50
      {51, 101},  // buckets 51-100
      {101, 201}, // buckets 101-200
      {201, 301}, // buckets 201-300
      {301, 401}, // buckets 301-400
      {401, 501}, // buckets 401-500
      {501, 601}, // buckets 501-600
      {601, 701}, // buckets 601-700
      {701, 801}, // buckets 701-800
      {801, 901}, // buckets 801-900
      {901, 1001}, // buckets 901-1000
      {1001, 1002} // overflow
    };

    long time_max_count = 0;
    for (int j = 0; j < 13; j++) {
      long total = 0;
      for (int k = bucket_ranges[j].start; k < bucket_ranges[j].end && k < HISTOGRAM_BUCKETS + 2; k++) {
        total += combined_time_histogram[k];
      }
      if (total > time_max_count) time_max_count = total;
    }

    if (time_max_count > 0) {
      for (int j = 0; j < 13; j++) {
        long total = 0;
        for (int k = bucket_ranges[j].start; k < bucket_ranges[j].end && k < HISTOGRAM_BUCKETS + 2; k++) {
          total += combined_time_histogram[k];
        }
        if (total == 0) continue;

        // Calculate the actual time range for this bucket range
        char label[50];
        if (bucket_ranges[j].start == 0) {
          strcpy(label, "≤0ns");
        } else if (bucket_ranges[j].start >= HISTOGRAM_BUCKETS + 1) {
          strcpy(label, "≥1s");
        } else {
          // Calculate time range using the same logarithmic formula as update()
          double log_max = log10((double)(MAX_DRAIN_TIME_NS + 1));

          // Start of range
          double log_min_start = (double)(bucket_ranges[j].start - 1) * log_max / HISTOGRAM_BUCKETS;
          long time_start = (long)(pow(10.0, log_min_start)) - 1;
          if (time_start < 0) time_start = 0;

          // End of range
          double log_min_end = (double)(bucket_ranges[j].end - 2) * log_max / HISTOGRAM_BUCKETS;
          long time_end = (long)(pow(10.0, log_min_end)) - 1;
          if (time_end >= MAX_DRAIN_TIME_NS) time_end = MAX_DRAIN_TIME_NS - 1;

          // Format the label with appropriate units
          if (time_end < 1000) {
            snprintf(label, sizeof(label), "%ldns-%ldns", time_start, time_end);
          } else if (time_end < 1000000) {
            snprintf(label, sizeof(label), "%.1fμs-%.1fμs", time_start/1000.0, time_end/1000.0);
          } else if (time_end < 1000000000) {
            snprintf(label, sizeof(label), "%.2fms-%.2fms", time_start/1000000.0, time_end/1000000.0);
          } else {
            snprintf(label, sizeof(label), "%.3fs-%.3fs", time_start/1000000000.0, time_end/1000000000.0);
          }
        }

        int bar_len = (int)((total * 40) / time_max_count);
        printf("%15s: ", label);
        for (int k = 0; k < bar_len; k++) printf("█");
        printf(" %ld\n", total);
      }
    }

    // ASCII visualization of combined event histogram
    printf("\nEvent Count Distribution:\n");
    long event_max_count = 0;
    for (int i = 0; i < MAX_EVENT_COUNT + 2; i++) {
      if (combined_event_histogram[i] > event_max_count) event_max_count = combined_event_histogram[i];
    }

    if (event_max_count > 0) {
      // Group events for display
      struct { int start, end; const char* label; } ranges[] = {
        {0, 1, "0"},
        {1, 6, "1-5"},
        {6, 11, "6-10"},
        {11, 21, "11-20"},
        {21, 51, "21-50"},
        {51, 101, "51-100"},
        {101, 201, "101-200"},
        {201, 501, "201-500"},
        {501, 1001, "501-1000"},
        {1001, 1002, ">1000"}
      };

      for (int j = 0; j < 10; j++) {
        long total = 0;
        for (int k = ranges[j].start; k < ranges[j].end && k < MAX_EVENT_COUNT + 2; k++) {
          total += combined_event_histogram[k];
        }
        if (total == 0) continue;

        int bar_len = (int)((total * 30) / event_max_count);
        printf("%8s: ", ranges[j].label);
        for (int k = 0; k < bar_len; k++) printf("█");
        printf(" %ld\n", total);
      }
    }

    // Machine readable combined JSON output
    stringStream combined_json_buffer;

    combined_json_buffer.print("COMBINED_DRAIN_STATS_JSON:{\"name\":\"%s\",\"drains\":%ld,\"time\":{\"sum\":%ld,\"avg\":%ld,\"min\":%ld,\"max\":%ld,\"median\":%ld,\"p90\":%ld,\"p95\":%ld,\"p99\":%ld,\"p99_9\":%ld},\"events\":{\"sum\":%ld,\"avg\":%.2f,\"min\":%ld,\"max\":%ld,\"median\":%ld,\"p90\":%ld,\"p95\":%ld,\"p99\":%ld,\"p99_9\":%ld},\"time_histogram\":[",
           name, total_drains, total_time_sum, total_drains > 0 ? total_time_sum / total_drains : 0, combined_time_min, combined_time_max,
           combined_time_median, combined_time_p90, combined_time_p95, combined_time_p99, combined_time_p999,
           total_event_sum, total_drains > 0 ? total_event_sum * 1.0 / total_drains : 0.0, combined_event_min, combined_event_max,
           combined_event_median, combined_event_p90, combined_event_p95, combined_event_p99, combined_event_p999);

    // Generate structured combined time histogram
    bool first_time = true;
    for (int i = 0; i < HISTOGRAM_BUCKETS + 2; i++) {
      if (combined_time_histogram[i] > 0) {
        if (!first_time) combined_json_buffer.print(",");
        first_time = false;

        if (i == 0) {
          combined_json_buffer.print("{\"from\":0,\"to\":1,\"count\":%ld,\"range\":\"underflow\"}", combined_time_histogram[i]);
        } else if (i >= HISTOGRAM_BUCKETS + 1) {
          combined_json_buffer.print("{\"from\":1000000000,\"to\":null,\"count\":%ld,\"range\":\"overflow\"}", combined_time_histogram[i]);
        } else {
          // Convert bucket back to time range
          double log_max = log10((double)(MAX_DRAIN_TIME_NS + 1));
          double log_min = (double)(i - 1) * log_max / HISTOGRAM_BUCKETS;
          double log_max_bucket = (double)i * log_max / HISTOGRAM_BUCKETS;
          long time_min_ns = (long)(pow(10.0, log_min)) - 1;
          long time_max_ns = (long)(pow(10.0, log_max_bucket)) - 1;
          if (time_min_ns < 0) time_min_ns = 0;
          if (time_max_ns >= MAX_DRAIN_TIME_NS) time_max_ns = MAX_DRAIN_TIME_NS - 1;
          combined_json_buffer.print("{\"from\":%ld,\"to\":%ld,\"count\":%ld}", time_min_ns, time_max_ns, combined_time_histogram[i]);
        }
      }
    }
    combined_json_buffer.print("],\"event_histogram\":[");

    // Generate structured combined event histogram
    bool first_event = true;
    for (int i = 0; i < MAX_EVENT_COUNT + 2; i++) {
      if (combined_event_histogram[i] > 0) {
        if (!first_event) combined_json_buffer.print(",");
        first_event = false;

        if (i >= MAX_EVENT_COUNT + 1) {
          combined_json_buffer.print("{\"from\":%ld,\"to\":null,\"count\":%ld,\"range\":\"overflow\"}", (long)MAX_EVENT_COUNT, combined_event_histogram[i]);
        } else {
          combined_json_buffer.print("{\"from\":%ld,\"to\":%ld,\"count\":%ld}", (long)i, (long)i, combined_event_histogram[i]);
        }
      }
    }
    combined_json_buffer.print("]}");

    // Output the complete combined JSON in one atomic operation
    tty->print_cr("%s", combined_json_buffer.base());
  }

  // Calculate percentiles from time histogram
  long calculate_time_percentile(double percentile) const {
    long total_samples = Atomic::load(&_drains);
    if (total_samples == 0) return 0;

    long target_count = (long)(total_samples * percentile / 100.0);
    long cumulative_count = 0;

    // Check underflow bucket first
    cumulative_count += Atomic::load(&_drain_time_histogram[0]);
    if (cumulative_count >= target_count) return 0;

    // Check regular buckets
    for (int i = 1; i <= HISTOGRAM_BUCKETS; i++) {
      cumulative_count += Atomic::load(&_drain_time_histogram[i]);
      if (cumulative_count >= target_count) {
        // Found the bucket containing the percentile
        // Convert bucket back to time value (middle of bucket range)
        double log_max = log10((double)(MAX_DRAIN_TIME_NS + 1));
        double log_mid = ((double)(i - 1) + 0.5) * log_max / HISTOGRAM_BUCKETS;
        long time_mid = (long)(pow(10.0, log_mid)) - 1;
        if (time_mid < 0) time_mid = 0;
        return time_mid;
      }
    }

    // If we get here, it's in the overflow bucket - return max value
    return MAX_DRAIN_TIME_NS;
  }

  // Calculate percentiles from event histogram
  long calculate_event_percentile(double percentile) const {
    long total_samples = Atomic::load(&_drains);
    if (total_samples == 0) return 0;

    long target_count = (long)(total_samples * percentile / 100.0);
    long cumulative_count = 0;

    // Check regular buckets (direct mapping for events)
    for (int i = 0; i <= MAX_EVENT_COUNT; i++) {
      cumulative_count += Atomic::load(&_event_histogram[i]);
      if (cumulative_count >= target_count) {
        return i; // Direct mapping: bucket index = event count
      }
    }

    // If we get here, it's in the overflow bucket
    return MAX_EVENT_COUNT + 1;
  }

  // Calculate combined percentiles from multiple DrainStats objects
  static long calculate_combined_time_percentile(const DrainStats& stats1, const DrainStats& stats2,
                                               const DrainStats& stats3, const DrainStats& stats4,
                                               double percentile) {
    long total_samples = stats1.count() + stats2.count() + stats3.count() + stats4.count();
    if (total_samples == 0) return 0;

    long target_count = (long)(total_samples * percentile / 100.0);
    long cumulative_count = 0;

    // Combine time histograms
    long combined_time_histogram[HISTOGRAM_BUCKETS + 2] = {0};
    const DrainStats* all_stats[] = {&stats1, &stats2, &stats3, &stats4};
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < HISTOGRAM_BUCKETS + 2; j++) {
        combined_time_histogram[j] += Atomic::load(&all_stats[i]->_drain_time_histogram[j]);
      }
    }

    // Check underflow bucket first
    cumulative_count += combined_time_histogram[0];
    if (cumulative_count >= target_count) return 0;

    // Check regular buckets
    for (int i = 1; i <= HISTOGRAM_BUCKETS; i++) {
      cumulative_count += combined_time_histogram[i];
      if (cumulative_count >= target_count) {
        // Convert bucket back to time value (middle of bucket range)
        double log_max = log10((double)(MAX_DRAIN_TIME_NS + 1));
        double log_mid = ((double)(i - 1) + 0.5) * log_max / HISTOGRAM_BUCKETS;
        long time_mid = (long)(pow(10.0, log_mid)) - 1;
        if (time_mid < 0) time_mid = 0;
        return time_mid;
      }
    }

    return MAX_DRAIN_TIME_NS;
  }

  static long calculate_combined_event_percentile(const DrainStats& stats1, const DrainStats& stats2,
                                                const DrainStats& stats3, const DrainStats& stats4,
                                                double percentile) {
    long total_samples = stats1.count() + stats2.count() + stats3.count() + stats4.count();
    if (total_samples == 0) return 0;

    long target_count = (long)(total_samples * percentile / 100.0);
    long cumulative_count = 0;

    // Combine event histograms
    long combined_event_histogram[MAX_EVENT_COUNT + 2] = {0};
    const DrainStats* all_stats[] = {&stats1, &stats2, &stats3, &stats4};
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < MAX_EVENT_COUNT + 2; j++) {
        combined_event_histogram[j] += Atomic::load(&all_stats[i]->_event_histogram[j]);
      }
    }

    // Check regular buckets
    for (int i = 0; i <= MAX_EVENT_COUNT; i++) {
      cumulative_count += combined_event_histogram[i];
      if (cumulative_count >= target_count) {
        return i;
      }
    }

    return MAX_EVENT_COUNT + 1;
  }
};

// Signal Handler Statistics tracking
struct SignalHandlerStats {
  static const int HISTOGRAM_BUCKETS = 1000;
  static const long MAX_HANDLER_TIME_NS = 100000000L; // 100ms in nanoseconds (reasonable max for signal handler)

  volatile long _signals_processed;
  volatile long _handler_time_sum;
  volatile long _handler_time_max;
  volatile long _handler_time_min;
  volatile long _last_print_time;
  volatile long _start_time;

  // Time histogram for signal handler duration: [0] = underflow, [1-1000] = buckets, [1001] = overflow
  volatile long _handler_time_histogram[HISTOGRAM_BUCKETS + 2];

  SignalHandlerStats() : _signals_processed(0), _handler_time_sum(0), _handler_time_max(0), _handler_time_min(LONG_MAX),
                        _last_print_time(0), _start_time(0) {
    for (int i = 0; i < HISTOGRAM_BUCKETS + 2; i++) {
      _handler_time_histogram[i] = 0;
    }
  }

  void update(long handler_duration_ns) {
    // Initialize start time on first update
    if (_start_time == 0) {
      _start_time = os::javaTimeNanos();
    }

    Atomic::inc(&_signals_processed);
    Atomic::add(&_handler_time_sum, handler_duration_ns);

    // Update handler time max
    if (handler_duration_ns > _handler_time_max) {
      while (true) {
        long old_max = _handler_time_max;
        if (handler_duration_ns <= old_max || Atomic::cmpxchg(&_handler_time_max, old_max, handler_duration_ns) == old_max) {
          break;
        }
      }
    }

    // Update handler time min
    if (handler_duration_ns < _handler_time_min) {
      while (true) {
        long old_min = _handler_time_min;
        if (handler_duration_ns >= old_min || Atomic::cmpxchg(&_handler_time_min, old_min, handler_duration_ns) == old_min) {
          break;
        }
      }
    }

    // Update handler time histogram (logarithmic bucketing)
    int time_bucket;
    if (handler_duration_ns <= 0) {
      time_bucket = 0; // underflow
    } else if (handler_duration_ns >= MAX_HANDLER_TIME_NS) {
      time_bucket = HISTOGRAM_BUCKETS + 1; // overflow
    } else {
      // Logarithmic bucketing: log10(handler_duration_ns + 1) mapped to [1, HISTOGRAM_BUCKETS]
      double log_time = log10((double)(handler_duration_ns + 1));
      double log_max = log10((double)(MAX_HANDLER_TIME_NS + 1));
      time_bucket = 1 + (int)((log_time * HISTOGRAM_BUCKETS) / log_max);
      if (time_bucket > HISTOGRAM_BUCKETS) time_bucket = HISTOGRAM_BUCKETS;
    }
    Atomic::inc(&_handler_time_histogram[time_bucket]);
  }

  bool should_print() {
    long current_time = os::javaTimeNanos();
    long last_print = Atomic::load(&_last_print_time);

    // Print every 1 second (1,000,000,000 ns)
    if (current_time - last_print >= 1000000000L) {
      // Try to update the last print time atomically
      if (Atomic::cmpxchg(&_last_print_time, last_print, current_time) == last_print) {
        return true; // We successfully updated the time, so we should print
      }
    }
    return false; // Either not enough time has passed or another thread is printing
  }

  // Calculate percentiles from signal handler duration histogram
  long calculate_handler_time_percentile(double percentile) const {
    long total_signals = Atomic::load(&_signals_processed);
    if (total_signals == 0) return 0;

    long target_count = (long)(total_signals * percentile / 100.0);
    long cumulative_count = 0;

    // Check underflow bucket first
    cumulative_count += Atomic::load(&_handler_time_histogram[0]);
    if (cumulative_count >= target_count) return 0;

    // Check regular buckets
    for (int i = 1; i <= HISTOGRAM_BUCKETS; i++) {
      cumulative_count += Atomic::load(&_handler_time_histogram[i]);
      if (cumulative_count >= target_count) {
        // Found the bucket containing the percentile
        // Convert bucket back to time value (middle of bucket range)
        double log_max = log10((double)(MAX_HANDLER_TIME_NS + 1));
        double log_mid = ((double)(i - 1) + 0.5) * log_max / HISTOGRAM_BUCKETS;
        long time_mid = (long)(pow(10.0, log_mid)) - 1;
        if (time_mid < 0) time_mid = 0;
        return time_mid;
      }
    }

    // If we get here, it's in the overflow bucket - return max value
    return MAX_HANDLER_TIME_NS;
  }

  void print(const char* name) {
    long signals = Atomic::load(&_signals_processed);
    long time_sum = Atomic::load(&_handler_time_sum);
    long time_max = Atomic::load(&_handler_time_max);
    long time_min = (_handler_time_min == LONG_MAX) ? 0 : Atomic::load(&_handler_time_min);

    if (signals == 0) {
      printf("Signal Handler Stats [%s]: No signals processed yet\n", name);
      return;
    }

    printf("Signal Handler Stats [%s]: signals=%ld avg=%.1fns min=%ldns max=%ldns\n",
           name, signals, signals > 0 ? time_sum * 1.0 / signals : 0.0, time_min, time_max);

    // Calculate percentiles
    long time_median = calculate_handler_time_percentile(50.0);
    long time_p90 = calculate_handler_time_percentile(90.0);
    long time_p95 = calculate_handler_time_percentile(95.0);
    long time_p99 = calculate_handler_time_percentile(99.0);
    long time_p999 = calculate_handler_time_percentile(99.9);

    printf("Signal Handler Percentiles [%s]: p50=%ldns p90=%ldns p95=%ldns p99=%ldns p99.9=%ldns\n",
           name, time_median, time_p90, time_p95, time_p99, time_p999);

    // Generate JSON output for machine parsing
    long current_time = os::javaTimeNanos();
    long runtime_ns = (_start_time > 0) ? (current_time - _start_time) : 0;
    double runtime_seconds = runtime_ns / 1000000000.0;

    stringStream json_buffer;
    json_buffer.print("SIGNAL_HANDLER_STATS_JSON:{\"name\":\"%s\",\"signals_processed\":%ld,\"runtime_ns\":%ld,\"runtime_seconds\":%.3f,\"handler_time\":{\"sum\":%ld,\"avg\":%ld,\"min\":%ld,\"max\":%ld,\"median\":%ld,\"p90\":%ld,\"p95\":%ld,\"p99\":%ld,\"p99_9\":%ld},\"handler_time_histogram\":[",
           name, signals, runtime_ns, runtime_seconds, time_sum, signals > 0 ? time_sum / signals : 0, time_min, time_max, time_median, time_p90, time_p95, time_p99, time_p999);

    // Generate structured time histogram
    bool first_time = true;
    for (int i = 0; i < HISTOGRAM_BUCKETS + 2; i++) {
      long count = Atomic::load(&_handler_time_histogram[i]);
      if (count > 0) {  // Only include non-zero buckets
        if (!first_time) json_buffer.print(",");
        first_time = false;

        if (i == 0) {
          json_buffer.print("{\"from\":0,\"to\":1,\"count\":%ld,\"range\":\"underflow\"}", count);
        } else if (i >= HISTOGRAM_BUCKETS + 1) {
          json_buffer.print("{\"from\":%ld,\"to\":null,\"count\":%ld,\"range\":\"overflow\"}", MAX_HANDLER_TIME_NS, count);
        } else {
          // Convert bucket back to time range
          double log_max = log10((double)(MAX_HANDLER_TIME_NS + 1));
          double log_start = ((double)(i - 1)) * log_max / HISTOGRAM_BUCKETS;
          double log_end = ((double)(i)) * log_max / HISTOGRAM_BUCKETS;
          long time_start = (long)(pow(10.0, log_start)) - 1;
          long time_end = (long)(pow(10.0, log_end)) - 1;
          if (time_start < 0) time_start = 0;
          if (time_end < 0) time_end = 0;
          json_buffer.print("{\"from\":%ld,\"to\":%ld,\"count\":%ld}", time_start, time_end, count);
        }
      }
    }

    json_buffer.print("]}\n");
    printf("%s", json_buffer.freeze());
    fflush(stdout);
  }
};

DrainStats out_of_thread_drain_stats;
DrainStats safepoint_drain_stats;
DrainStats safepoint_drain_stats_w_locking;
DrainStats drain_stats;

// Signal handler statistics instance
SignalHandlerStats signal_handler_stats;

// Internal function to print signal handler stats periodically
static void print_signal_handler_stats_if_needed() {
  if (signal_handler_stats.should_print()) {
    signal_handler_stats.print("signal_handler");
  }
}

static void drain_enqueued_cpu_time_requests(const JfrTicks& now, JfrThreadLocal* tl, JavaThread* jt, Thread* current, bool lock) {
  assert(tl != nullptr, "invariant");
  assert(jt != nullptr, "invariant");
  assert(current != nullptr, "invariant");

#ifdef LINUX
  tl->set_do_async_processing_of_cpu_time_jfr_requests(false);
  long startWithLock = os::javaTimeNanos();
  if (lock) {
    tl->acquire_cpu_time_jfr_dequeue_lock();
  }
    long start = os::javaTimeNanos();

  JfrCPUTimeTraceQueue& queue = tl->cpu_time_jfr_queue();
  long size = queue.size();

  for (u4 i = 0; i < queue.size(); i++) {
    record_cpu_time_thread(queue.at(i), now, tl, jt, current);
  }
  queue.clear();
  assert(queue.is_empty(), "invariant");
  tl->set_has_cpu_time_jfr_requests(false);
  if (queue.lost_samples() > 0) {
    JfrCPUTimeThreadSampling::send_lost_event( now, JfrThreadLocal::thread_id(jt), queue.get_and_reset_lost_samples());
  }
      long time = os::javaTimeNanos() - start;

  drain_stats.update(os::javaTimeNanos() - start, size);
  if (lock) {
    safepoint_drain_stats.update(os::javaTimeNanos() - start, size);
    tl->release_cpu_time_jfr_queue_lock();
  }
  if (lock) {
        safepoint_drain_stats_w_locking.update(os::javaTimeNanos() - startWithLock, size);

  } else {
        out_of_thread_drain_stats.update(os::javaTimeNanos() - start, size);

  }
  // Print individual drain category statistics every 30 seconds
  if (drain_stats.should_print()) {
    // Check if STATS_FILE environment variable is set
    const char* stats_file = ::getenv("STATS_FILE");
    if (stats_file != nullptr) {
      // Thread-safe file writing using PlatformMutex
      static PlatformMutex stats_file_lock;

      // RAII lock guard
      class LockGuard {
        PlatformMutex* _mutex;
      public:
        LockGuard(PlatformMutex* mutex) : _mutex(mutex) { _mutex->lock(); }
        ~LockGuard() { _mutex->unlock(); }
      };
      LockGuard lock_guard(&stats_file_lock);

      // Open file in append mode
      FILE* file = fopen(stats_file, "a");
      if (file != nullptr) {
        // Temporarily redirect stdout to file
        FILE* original_stdout = stdout;
        stdout = file;

        drain_stats.print("all without locks");
        safepoint_drain_stats.print("safepoint");
        safepoint_drain_stats_w_locking.print("safepoint with locks");
        out_of_thread_drain_stats.print("out of thread");
        signal_handler_stats.print("signal handler duration");

        // Restore stdout and close file
        stdout = original_stdout;
        fclose(file);
      } else {
        // Fall back to stdout if file can't be opened
        drain_stats.print("all without locks");
        safepoint_drain_stats.print("safepoint");
        safepoint_drain_stats_w_locking.print("safepoint with locks");
        out_of_thread_drain_stats.print("out of thread");
        signal_handler_stats.print("signal handler duration");
      }
    } else {
      // Default behavior - print to stdout
      drain_stats.print("all without locks");
      safepoint_drain_stats.print("safepoint");
      safepoint_drain_stats_w_locking.print("safepoint with locks");
      out_of_thread_drain_stats.print("out of thread");
      signal_handler_stats.print("signal handler duration");
    }
  }
#endif
}

// Entry point for a thread that has been sampled in native code and has a pending JFR CPU time request.
void JfrThreadSampling::process_cpu_time_request(JavaThread* jt, JfrThreadLocal* tl, Thread* current, bool lock) {
  assert(jt != nullptr, "invariant");
  const JfrTicks now = JfrTicks::now();
  drain_enqueued_cpu_time_requests(now, tl, jt, current, lock);
}

static void drain_all_enqueued_requests(const JfrTicks& now, JfrThreadLocal* tl, JavaThread* jt, Thread* current) {
  assert(tl != nullptr, "invariant");
  assert(jt != nullptr, "invariant");
  assert(current != nullptr, "invariant");
  drain_enqueued_requests(now, tl, jt, current);
  if (tl->has_cpu_time_jfr_requests()) {
    drain_enqueued_cpu_time_requests(now, tl, jt, current, true);
  }
}

// Only entered by the JfrSampler thread.
bool JfrThreadSampling::process_native_sample_request(JfrThreadLocal* tl, JavaThread* jt, Thread* sampler_thread) {
  assert(tl != nullptr, "invairant");
  assert(jt != nullptr, "invariant");
  assert(sampler_thread != nullptr, "invariant");
  assert(sampler_thread->is_JfrSampler_thread(), "invariant");
  assert(tl == jt->jfr_thread_local(), "invariant");
  assert(jt != sampler_thread, "only asynchronous processing of native samples");
  assert(jt->has_last_Java_frame(), "invariant");
  assert(tl->sample_state() >= NATIVE_SAMPLE, "invariant");

  assert_lock_strong(Threads_lock);

  const JfrTicks start_time = JfrTicks::now();

  traceid tid;
  traceid sid;

  {
    JfrSampleMonitor sm(tl);

    // Because the thread was in native, it is in a walkable state, because
    // it will hit a safepoint poll on the way back from native. To ensure timely
    // progress, any requests in the queue can be safely processed now.
    drain_enqueued_requests(start_time, tl, jt, sampler_thread);
    // Process the current stacktrace using the ljf.
    {
      ResourceMark rm(sampler_thread);
      JfrStackTrace stacktrace;
      const frame top_frame = jt->last_frame();
      if (!stacktrace.record_inner(jt, top_frame, is_in_continuation(top_frame, jt), 0 /* skip level */)) {
        // Unable to record stacktrace. Fail.
        return false;
      }
      sid = JfrStackTraceRepository::add(stacktrace);
    }
    // Read the tid under the monitor to ensure that if its a virtual thread,
    // it is not unmounted until we are done with it.
    tid = JfrThreadLocal::thread_id(jt);
  }

  assert(tl->sample_state() == NO_SAMPLE, "invariant");
  send_sample_event<EventNativeMethodSample>(start_time, start_time, sid, tid);
  return true;
}

// Entry point for a sampled thread that discovered pending Jfr Sample Requests as part of a safepoint poll.
void JfrThreadSampling::process_sample_request(JavaThread* jt) {
  assert(JavaThread::current() == jt, "should be current thread");
  assert(jt->thread_state() == _thread_in_vm || jt->thread_state() == _thread_in_Java, "invariant");

  const JfrTicks now = JfrTicks::now();

  JfrThreadLocal* const tl = jt->jfr_thread_local();
  assert(tl != nullptr, "invariant");

  MonitorLocker ml(tl->sample_monitor(), Monitor::_no_safepoint_check_flag);

  for (;;) {
    const int sample_state = tl->sample_state();
    if (sample_state == NATIVE_SAMPLE) {
      tl->set_sample_state(WAITING_FOR_NATIVE_SAMPLE);
      // Wait until stack trace is processed.
      ml.wait();
    } else if (sample_state == JAVA_SAMPLE) {
      tl->enqueue_request();
    } else if (sample_state == WAITING_FOR_NATIVE_SAMPLE) {
      // Handle spurious wakeups. Again wait until stack trace is processed.
      ml.wait();
    } else {
      // State has been processed.
      break;
    }
  }
  drain_all_enqueued_requests(now, tl, jt, jt);
}

// Signal handler statistics public methods
void JfrThreadSampling::record_signal_handler_duration(long duration_ns) {
  signal_handler_stats.update(duration_ns);
  // Call the internal print function
  print_signal_handler_stats_if_needed();
}

