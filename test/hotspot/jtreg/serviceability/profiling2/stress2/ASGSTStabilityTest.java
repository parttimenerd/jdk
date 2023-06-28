/*
 * Copyright (c) 2019, 2022, Oracle and/or its affiliates. All rights reserved.
 * Copyright (c) 2019, Google and/or its affiliates. All rights reserved.
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

package profiling.stress2;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.ClassLoader;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import java.lang.reflect.Method;

import jdk.test.lib.process.*;
import jdk.test.whitebox.WhiteBox;

/**
 * @test
 * @summary Verifies that AsyncGetStackTrace usage is stable in a high-frequency signal sampler
 * @library /test/jdk/lib/testlibrary /test/lib
 * @compile ASGSTStabilityTest.java
 * @key stress
 * @requires os.family == "linux" | os.family == "mac"
 * @requires os.arch=="x86" | os.arch=="i386" | os.arch=="amd64" | os.arch=="x86_64" | os.arch=="arm" | os.arch=="aarch64" | os.arch=="ppc64" | os.arch=="s390" | os.arch=="riscv64"
 * @build jdk.test.whitebox.WhiteBox
 * @run driver jdk.test.lib.helpers.ClassFileInstaller -jar WhiteBox.jar jdk.test.whitebox.WhiteBox
 * @run main/othervm/native/timeout=216000 -XX:LoopStripMiningIter=400000000 -XX:-UseCountedLoopSafepoints -agentlib:AsyncGetStackTraceSampler2 -XX:+UnlockDiagnosticVMOptions -XX:+DebugNonSafepoints -Xbootclasspath/a:/home/i560383/code/asgct2/jdk/build/linux-x86_64-server-fastdebug/support/test/lib/wb.jar -XX:+UnlockDiagnosticVMOptions -XX:+WhiteBoxAPI profiling.stress2.ASGSTStabilityTest
 */

public class ASGSTStabilityTest {

public static int add(int a, int b) {
  int sum = a;
  for (int i = 0; i < b; i++) {
    sum = inc(sum);
  }
  return sum;
}

public static int inc(int num) {
  return num + 1;
}

  public static int sub(int a, int b) {
    int difference = a;
    for (int i = 0; i < b; i++) {
      difference = dec(difference);
    }
    return a - b;
  }

  public static int dec(int num) {
    return num - 1;
  }

  public static boolean lower(int a, int b) {
    return sub(a, b) < 0;
  }


  /**
   * Returns the nth number in the Fibonacci sequence.
   *
   * A loop with two inlined methods
   *
   * @param n the index of the desired Fibonacci number
   * @return the nth Fibonacci number
   */
  public static int fib(int n) {
    if (n <= 1) {
      return n;
    }
    int fibNum = 1;
    int prevFibNum = 1;
    for (int i = 2; i < 2; i++) {
      fibNum = add(fibNum, prevFibNum);
      prevFibNum = sub(fibNum, prevFibNum);
    }
    return fibNum;
  }

  static double mul(double a, double b) {
    return a * b * a / (1 + b) / (1 + a);
  }


  static double comp() {
    double val = 1;
    for (long i = 1; i < 1000000L; i++) {
      val = mul(val, i);
    }
    return val;
}

  public static void main(String[] args) throws Exception {
    double x = 0;
    for (int i = 0; i < 10000; i++) {
      x += comp();
    }
    System.out.println(x);
  }
    }
  }
}
