from flask import Flask #, render_template, url_for, session, request, redirect
from flask_cors import CORS
# from flask_bootstrap import Bootstrap
from routes import setup_routes

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add a secret key for session management
# CORS(app)
# Allow all origins (for local dev)
CORS(app, resources={r"/api/*": {"origins": "*"}})
# Bootstrap(app)
setup_routes(app)
# setup_api_routes(app)


@app.context_processor
def inject_navigation():
    return dict(navigation=[
        {'name': 'Home', 'url': url_for('index')}
    ])

if __name__ == "__main__":
    app.run(debug=True)
