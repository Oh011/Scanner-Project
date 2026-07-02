#  Running the  Vulnerability Scanner

Follow the steps below to run the project locally.

## 1. Clone the Repository

```bash
git clone <repository-url>
cd AI-VulScanner
```

---

## 2. Install Dependencies



```bash
pip install -r requirements.txt
```

---

## 3. Start the FastAPI Backend

Navigate to the `src` directory:

```bash
cd src
```

Run the API using Uvicorn:

```bash
uvicorn Api:app --reload
```

The API will start at:

```
http://127.0.0.1:8000
```

You can also access the Swagger documentation at:

```
http://127.0.0.1:8000/docs
```

---

## 4. Open the Web Application

Return to the project root directory.

You will find a folder named:

```
Web App/
```

Open the HTML file inside this folder (for example, `DashBoard.html`) using your browser.

> **Note:** Make sure the FastAPI backend is running before opening the web application, as the HTML page sends requests to the API.

---


---

