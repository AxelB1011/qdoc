import unittest
from app.services import query_data

class TestQuery(unittest.TestCase):
    def test_query(self):
        request = {"query": "What is Llama 2?"}
        response = query_data(request)
        self.assertIn("response", response)

if __name__ == "__main__":
    unittest.main()
