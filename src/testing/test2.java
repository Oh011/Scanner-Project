import java.io.*;

public class CommandRunner {

    public void run(String userInput) throws Exception {

        String command =
                "ping " + userInput;

        Process process =
                Runtime.getRuntime().exec(command);

        BufferedReader reader =
                new BufferedReader(
                        new InputStreamReader(
                                process.getInputStream()
                        )
                );

        String line;

        while ((line = reader.readLine()) != null) {
            System.out.println(line);
        }
    }
}