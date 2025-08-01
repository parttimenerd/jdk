import java.util.concurrent.*;

public class CPUStressTest {
    static { System.loadLibrary("cpustress"); }

    public native void work(int threadId, int seconds);

    public static void main(String[] args) throws Exception {
        int duration = args.length > 0 ? Integer.parseInt(args[0]) : 10;
        int cores = Runtime.getRuntime().availableProcessors();

        System.out.println("Running " + duration + "s on " + cores + " cores");

        ExecutorService exec = Executors.newFixedThreadPool(cores);
        CPUStressTest test = new CPUStressTest();

        for (int i = 0; i < cores; i++) {
            final int id = i;
            exec.submit(() -> test.work(id, duration));
        }

        exec.shutdown();
        exec.awaitTermination(duration + 10, TimeUnit.SECONDS);
        System.out.println("Done");
    }
}