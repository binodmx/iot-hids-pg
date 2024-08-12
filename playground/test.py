from ragml import RAGML
import os
import dotenv

dotenv.load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

ragml = RAGML(API_KEY)
# ragml.summary()
# ragml.predict("ice cream flavors")
ragml.fit(["data packet", "data packet", "data packt"], ["normal", "normal", "anomaly"])
ragml.predict_one("data packett")
ragml.predict_one("data packet")