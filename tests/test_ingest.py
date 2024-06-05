import unittest
from app.services import ingest_data

class TestIngest(unittest.TestCase):
    def test_ingest(self):
        request = {"file_path": "data/llama2.pdf"}
        response = ingest_data(request)
        self.assertEqual(response["status"], "success")

if __name__ == "__main__":
    unittest.main()
