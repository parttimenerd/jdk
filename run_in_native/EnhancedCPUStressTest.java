import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.lang.reflect.Method;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

/**
 * Enhanced CPU Stress Test with adjustable stack sizes and dynamic class generation.
 * This creates maximum complexity for the JFR CPU time sampler by:
 * 1. Generating new classes dynamically at runtime
 * 2. Creating deep call stacks with configurable depth
 * 3. Mixing Java and native code execution
 * 4. Cleaning up generated classes after use
 */
public class EnhancedCPUStressTest {
    static {
        try {
            System.loadLibrary("cpustress");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Warning: Native library 'cpustress' not found. Native operations will be skipped.");
        }
    }

    // Native method for CPU-intensive work
    public native void work(int threadId, int seconds, int stackDepth);

    // Configuration
    private static final String TEMP_CLASS_PREFIX = "DynamicStressClass";
    private static final String TEMP_DIR = System.getProperty("java.io.tmpdir") + "/jfr_stress_test/";
    private static final AtomicInteger classCounter = new AtomicInteger(0);
    private static final List<String> generatedClassFiles = new ArrayList<>();
    private static final Random random = new Random();

    // Stack depth management
    private int currentStackDepth = 0;
    private final int maxStackDepth;
    private final boolean useNative;

    public EnhancedCPUStressTest(int maxStackDepth, boolean useNative) {
        this.maxStackDepth = maxStackDepth;
        this.useNative = useNative;
        setupTempDirectory();
    }

    /**
     * Recursively builds stack depth before calling native work function
     */
    public void buildStackAndWork(int threadId, int seconds, int targetDepth) {
        if (currentStackDepth >= targetDepth) {
            // We've reached the target depth, now do the actual work
            if (useNative) {
                try {
                    work(threadId, seconds, currentStackDepth);
                } catch (UnsatisfiedLinkError e) {
                    // Fallback to Java-only implementation
                    doJavaWork(threadId, seconds);
                }
            } else {
                doJavaWork(threadId, seconds);
            }
        } else {
            // Build more stack depth
            currentStackDepth++;

            // Occasionally use dynamically generated classes to make things interesting
            if (random.nextInt(10) == 0) {
                try {
                    callDynamicClass(threadId, seconds, targetDepth);
                } catch (Exception e) {
                    // Fallback to regular recursion
                    buildStackAndWork(threadId, seconds, targetDepth);
                }
            } else {
                // Regular recursion with some computation to make frames interesting
                double dummy = performSomeComputation(threadId, currentStackDepth);
                buildStackAndWork(threadId, seconds, targetDepth);
            }
            currentStackDepth--;
        }
    }

    /**
     * Creates and calls a dynamically generated class
     */
    private void callDynamicClass(int threadId, int seconds, int targetDepth) throws Exception {
        String className = TEMP_CLASS_PREFIX + classCounter.incrementAndGet();
        String javaCode = generateDynamicClassCode(className);

        // Compile the class
        String javaFile = TEMP_DIR + className + ".java";
        String classFile = TEMP_DIR + className + ".class";

        synchronized (generatedClassFiles) {
            generatedClassFiles.add(javaFile);
            generatedClassFiles.add(classFile);
        }

        // Write source file
        try (FileWriter writer = new FileWriter(javaFile)) {
            writer.write(javaCode);
        }

        // Compile
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        int result = compiler.run(null, null, null, "-d", TEMP_DIR, javaFile);

        if (result == 0) {
            // Load and execute the class
            URLClassLoader loader = new URLClassLoader(new URL[]{new File(TEMP_DIR).toURI().toURL()});
            Class<?> dynamicClass = loader.loadClass(className);
            Method method = dynamicClass.getMethod("continueStressTest",
                EnhancedCPUStressTest.class, int.class, int.class, int.class);

            method.invoke(null, this, threadId, seconds, targetDepth);
            loader.close();
        } else {
            // Compilation failed, fallback
            buildStackAndWork(threadId, seconds, targetDepth);
        }
    }

    /**
     * Generates source code for a dynamic stress test class
     */
    private String generateDynamicClassCode(String className) {
        return String.format("""
            public class %s {
                public static void continueStressTest(EnhancedCPUStressTest test, int threadId, int seconds, int targetDepth) {
                    // Add some unique computation per class
                    double result = 0.0;
                    for (int i = 0; i < %d; i++) {
                        result += Math.sin(i * 0.001) * Math.cos(threadId * 0.002);
                    }

                    // Continue building the stack
                    test.buildStackAndWork(threadId, seconds, targetDepth);

                    // More computation on the way back up
                    for (int i = 0; i < %d; i++) {
                        result += Math.exp(result * 0.0001) - Math.log(Math.abs(result) + 1);
                        if (result > 1000) result = 1.0;
                    }
                }
            }
            """,
            className,
            1000 + random.nextInt(5000),  // Variable computation load
            500 + random.nextInt(2000)
        );
    }

    /**
     * Performs some computation to make stack frames more interesting
     */
    private double performSomeComputation(int threadId, int stackDepth) {
        double result = threadId + stackDepth;
        for (int i = 0; i < 100; i++) {
            result = Math.sin(result) + Math.cos(result * 1.1) + Math.sqrt(Math.abs(result));
            if (result > 1000) result = 1.0;
        }
        return result;
    }

    /**
     * Pure Java CPU-intensive work as fallback
     */
    private void doJavaWork(int threadId, int seconds) {
        long endTime = System.currentTimeMillis() + (seconds * 1000L);
        double x = threadId + 1.0;

        while (System.currentTimeMillis() < endTime) {
            for (int i = 0; i < 10000; i++) {
                x = Math.sin(x) + Math.cos(x * 1.1) + Math.sqrt(Math.abs(x));
                if (x > 1000) x = 1.0;

                // Add some object allocation to stress GC too
                if (i % 1000 == 0) {
                    String dummy = "stress_" + threadId + "_" + i;
                    dummy.hashCode(); // Use it
                }
            }
        }
    }

    /**
     * Set up temporary directory for generated classes
     */
    private void setupTempDirectory() {
        try {
            Path tempPath = Paths.get(TEMP_DIR);
            Files.createDirectories(tempPath);
        } catch (IOException e) {
            System.err.println("Warning: Could not create temp directory for dynamic classes: " + e.getMessage());
        }
    }

    /**
     * Clean up generated class files
     */
    public static void cleanup() {
        synchronized (generatedClassFiles) {
            for (String filename : generatedClassFiles) {
                try {
                    Files.deleteIfExists(Paths.get(filename));
                } catch (IOException e) {
                    // Ignore cleanup errors
                }
            }
            generatedClassFiles.clear();
        }

        // Try to remove the temp directory
        try {
            Files.deleteIfExists(Paths.get(TEMP_DIR));
        } catch (IOException e) {
            // Ignore cleanup errors
        }
    }

    /**
     * Main method with enhanced configuration options
     */
    public static void main(String[] args) throws Exception {
        // Parse arguments
        int duration = args.length > 0 ? Integer.parseInt(args[0]) : 10;
        int cores = args.length > 1 ? Integer.parseInt(args[1]) : Runtime.getRuntime().availableProcessors();
        int minStackDepth = args.length > 2 ? Integer.parseInt(args[2]) : 10;
        int maxStackDepth = args.length > 3 ? Integer.parseInt(args[3]) : 50;
        boolean useNative = args.length > 4 ? Boolean.parseBoolean(args[4]) : true;

        System.out.println("Enhanced CPU Stress Test Configuration:");
        System.out.println("  Duration: " + duration + " seconds");
        System.out.println("  Threads: " + cores);
        System.out.println("  Stack depth range: " + minStackDepth + " - " + maxStackDepth);
        System.out.println("  Use native code: " + useNative);
        System.out.println("  Dynamic class generation: enabled");

        // Add shutdown hook for cleanup
        Runtime.getRuntime().addShutdownHook(new Thread(EnhancedCPUStressTest::cleanup));

        ExecutorService executor = Executors.newFixedThreadPool(cores);
        List<Future<?>> futures = new ArrayList<>();

        // Start stress test threads
        for (int i = 0; i < cores; i++) {
            final int threadId = i;
            Future<?> future = executor.submit(() -> {
                try {
                    // Each thread uses a different stack depth
                    int stackDepth = minStackDepth + (threadId % (maxStackDepth - minStackDepth + 1));
                    EnhancedCPUStressTest test = new EnhancedCPUStressTest(stackDepth, useNative);

                    System.out.println("Thread " + threadId + " starting with stack depth " + stackDepth);
                    test.buildStackAndWork(threadId, duration, stackDepth);
                    System.out.println("Thread " + threadId + " completed");

                } catch (Exception e) {
                    System.err.println("Thread " + threadId + " failed: " + e.getMessage());
                    e.printStackTrace();
                }
            });
            futures.add(future);
        }

        // Wait for completion
        executor.shutdown();

        try {
            if (!executor.awaitTermination(duration + 30, TimeUnit.SECONDS)) {
                System.err.println("Timeout waiting for threads to complete");
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            System.err.println("Interrupted while waiting for completion");
            executor.shutdownNow();
        }

        // Check for any failures
        for (int i = 0; i < futures.size(); i++) {
            try {
                futures.get(i).get();
            } catch (Exception e) {
                System.err.println("Thread " + i + " threw exception: " + e.getMessage());
            }
        }

        // Cleanup
        cleanup();
        System.out.println("Enhanced CPU stress test completed and cleaned up");
    }
}
