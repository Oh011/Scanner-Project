import java.io.*;
import javax.servlet.http.*;

public class ProfileServlet extends HttpServlet {

    protected void doGet(
            HttpServletRequest request,
            HttpServletResponse response)
            throws IOException {

        String name =
                request.getParameter("name");

        PrintWriter out =
                response.getWriter();

        out.println("<html>");
        out.println("<body>");
        out.println("<h2>" + name + "</h2>");
        out.println("</body>");
        out.println("</html>");
    }
}