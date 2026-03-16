from http.server import BaseHTTPRequestHandler
import json
from urllib.parse import parse_qs, urlparse

from app import get_predictions


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        tickers = query.get("tickers", ["AAPL,MSFT,NVDA,TSLA,AMZN"])[0]
        period = query.get("period", ["6mo"])[0]

        try:
            result = get_predictions(tickers=tickers, period=period)
            body = json.dumps(result, default=str)
            status = 200
        except Exception as e:
            body = json.dumps({"error": str(e)})
            status = 500

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))
