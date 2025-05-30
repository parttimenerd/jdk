/*
 * Copyright (c) 2025, Rivos Inc. All rights reserved.
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
 */

package org.openjdk.bench.java.lang;

import org.openjdk.jmh.annotations.*;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.random.RandomGenerator;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(3)
public class PointerComparison {
    static final int INVOCATIONS = 1024;

    Object[] o1;
    Object[] o2;
    int[] res;
    long[] resLong;
    Object[] resObject;
    Object ro1;
    Object ro2;
    Object[] resClass;
    Class rc1;
    Class rc2;

    @Setup
    public void setup() {
        var random = RandomGenerator.getDefault();
        o1 = new Object[INVOCATIONS];
        o2 = new Object[INVOCATIONS];
        res = new int[INVOCATIONS];
        resLong = new long[INVOCATIONS];
        resObject = new Object[INVOCATIONS];
        ro1 = new Object();
        ro2 = new Object();
        resClass = new Class[INVOCATIONS];
        rc1 = Float.class;
        rc2 = Double.class;
        for (int i = 0; i < INVOCATIONS; i++) {
            o1[i] = new Object();
        }
        List<Object> list = Arrays.asList(o1);
        Collections.shuffle(list);
        list.toArray(o2);
    }

    @Benchmark
    public void equalObject() {
        for (int i = 0; i < INVOCATIONS; i++) {
            res[i] = (o1[i] == o2[i]) ? 1 : 2;
        }
    }

    @Benchmark
    public void notEqualObject() {
        for (int i = 0; i < INVOCATIONS; i++) {
            res[i] = (o1[i] != o2[i]) ? 1 : 2;
        }
    }

    public void equalObjectResLong() {
        for (int i = 0; i < INVOCATIONS; i++) {
            resLong[i] = (o1[i] == o2[i]) ? Long.MAX_VALUE : Long.MIN_VALUE;
        }
    }

    @Benchmark
    public void notEqualObjectResLong() {
        for (int i = 0; i < INVOCATIONS; i++) {
            resLong[i] = (o1[i] != o2[i]) ? Long.MAX_VALUE : Long.MIN_VALUE;
        }
    }
}
