import traceback
from runner import LeanRunner

runner = LeanRunner()
try:
    runner.process_all_accepted_alphas()
except Exception as e:
    traceback.print_exc()
