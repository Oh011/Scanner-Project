import java.sql.*;
import javax.servlet.http.*;

public class LoginServlet {

    public void login(HttpServletRequest request) throws Exception {

        String username = request.getParameter("username");

        Connection conn = DriverManager.getConnection(
                "jdbc:mysql://localhost/test",
                "root",
                "password"
        );

        Statement stmt = conn.createStatement();

        String query =
                "SELECT * FROM users WHERE username='" +
                        username +
                        "'";

        ResultSet rs = stmt.executeQuery(query);

        while (rs.next()) {
            System.out.println(rs.getString("username"));
        }
    }
}