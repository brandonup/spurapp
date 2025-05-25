# Spurcycle Sales-Insight App

This application provides sales insights based on uploaded CSV data.

## Local Development

### Prerequisites
- Python 3.11
- pip

### Setup
1. Clone the repository.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your OpenAI API Key:
   - For local development, you can create a `.env` file in the project root with the following content:
     ```
     OPENAI_API_KEY='your_openai_api_key_here'
     ```
   - The application will load this key if `python-dotenv` is installed and used in `app.py`.

### Running the App
```bash
streamlit run app.py
```

## Deployment to Vercel

1. Push your project to a Git repository (e.g., GitHub, GitLab, Bitbucket).
2. Import your Git repository into Vercel.
3. Configure the Project Settings in Vercel:
   - **Build & Development Settings**:
     - FRAMEWORK PRESET: `Other`
     - BUILD COMMAND: (Leave empty or use `pip install -r requirements.txt` if Vercel doesn't do it automatically)
     - OUTPUT DIRECTORY: (Leave as default)
     - INSTALL COMMAND: `pip install -r requirements.txt streamlit`
   - **Environment Variables**:
     - Add `OPENAI_API_KEY` with your OpenAI API key.
4. Create a `vercel.json` file in the root of your project with the following content:
   ```json
   {
     "builds": [
       {
         "src": "app.py",
         "use": "@vercel/python",
         "config": { "maxLambdaSize": "15mb", "runtime": "python3.11" }
       }
     ],
     "routes": [
       {
         "src": "/(.*)",
         "dest": "app.py"
       }
     ]
   }
   ```
   *Note: The Vercel Python runtime might require specific configurations. The above is a general guideline. You might need to adjust it based on Vercel's current Python support for Streamlit apps. An alternative approach for `vercel.json` if running Streamlit directly as a service:*
   ```json
   {
     "functions": {
       "api/index.py": {
         "runtime": "python3.11"
       }
     },
     "routes": [{ "src": "/(.*)", "dest": "/api/index.py" }]
   }
   ```
   *And create `api/index.py`:*
   ```python
   import os
   import subprocess

   def handler(request, response):
       port = os.environ.get("PORT", "8501")
       process = subprocess.Popen(
           ["streamlit", "run", "app.py", "--server.port", port, "--server.headless", "true"],
           stdout=subprocess.PIPE,
           stderr=subprocess.PIPE,
       )
       stdout, stderr = process.communicate()
       
       # This is a simplified handler. For a production app, 
       # you'd likely proxy requests to the Streamlit server.
       # Vercel's Python runtime might not be ideal for long-running Streamlit apps directly.
       # Consider Streamlit Community Cloud for simpler deployment.
       
       response.status_code = 200
       response.send(f"Streamlit app output (stdout):\n{stdout.decode()}\n\nStderr:\n{stderr.decode()}")
   ```

5. Deploy.

### Alternative: Streamlit Community Cloud
If Vercel deployment proves complex or flaky for Streamlit apps, consider deploying to [Streamlit Community Cloud](https://streamlit.io/cloud), which is optimized for Streamlit apps.
