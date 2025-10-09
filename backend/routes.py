# import pandas as pd
# import numpy as np
# import io
# import os
from flask import request, jsonify
# import json
# import requests
# from io import StringIO
# from flask_cors import CORS
# import traceback

# from stock_data import StockData

# DEBUG = True
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# # DATA_PATH = os.path.join(BASE_DIR, "data", "feature_sets.json")
# # FEATURE_SETS_FILE = os.path.join(os.path.dirname(__file__), 'data\\feature_sets.json')
# FEATURE_SETS_FILE = os.path.join(BASE_DIR, "back-end", "data", "feature_sets.json")


# API Routes for Front-End
def setup_api_routes(app):
    """
    Set up all routes for the Flask application.

    Args:
        app (Flask): The Flask application instance.
    """

    @app.route("/api/upload_data", methods=["POST", "OPTIONS"])
    def upload_data():
        if request.method == "OPTIONS":
            # Preflight request handled automatically by flask-cors
            return jsonify({"status": "ok"}), 200

        if "fileName" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        fileName = request.files['fileName']
        
        try:
            raw_data = load_file(fileName)

            response = {
                "success": True,
                "timeSeriesData": raw_data,
            }

            return jsonify(response)

        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            print(f"Upload error: {error_msg}")
            # print(traceback.format_exc())
            return jsonify({'error': error_msg}), 500

    @app.route("/api/yahoo_data", methods=["POST", "OPTIONS"])
    def yahoo_data():
        if request.method == "OPTIONS":
            # Preflight request handled automatically by flask-cors
            return jsonify({"status": "ok"}), 200

        data = request.get_json()
        ticker = data.get("ticker")
        start_date = data.get("start_date")
        end_date = data.get("end_date")

        print(data)

        if not ticker:
            return jsonify({"success": False, "error": "Ticker is required"}), 400

        try:
            df = yf.download(tickers=ticker, start=start_date, end=end_date, auto_adjust=False)
            print(df)
            if df.empty:
                return jsonify({"success": False, "error": "No data found"}), 404

            df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
            df = df.reset_index()

            return jsonify({
                "success": True,
                "timeSeriesData": df.to_dict(orient="records"),
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/get_tickers/<category>', methods=['GET'])
    # @app.route('/api/get_tickers/<category>', methods=['GET', "OPTIONS"])
    def get_tickers(category):
        """
        Fetch tickers for the given category.

        Args:
            category (str): The selected category.

        Returns:
            Response: JSON response containing the list of tickers.
        """
        # if request.method == "OPTIONS":
        #     # Preflight request handled automatically by flask-cors
        #     return jsonify({"status": "ok"}), 200
            
        print(f"Getting tickers for {category}")
        tickers = get_tickers_by_category(category)
        return jsonify({'tickers': tickers})

    @app.route("/api/feature_sets", methods=["GET", "OPTIONS"])
    def get_feature_sets():
        if request.method == "OPTIONS":
            # Preflight request handled automatically by flask-cors
            return jsonify({"status": "ok"}), 200

        print(FEATURE_SETS_FILE)
        try:
            with open(FEATURE_SETS_FILE, "r") as f:
                feature_sets = json.load(f)
            return jsonify(feature_sets), 200
        except FileNotFoundError:
            return jsonify({"error": "feature_sets.json not found"}), 404
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format"}), 500


    # @app.route("/api/functions", methods=["GET"])
    # def get_functions():
