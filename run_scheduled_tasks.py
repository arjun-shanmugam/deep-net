import schedule
import time
from utils.send_email import send_email
 
def daily_tasks():
    #TODO run daily tasks
    send_email(["1"])

def run():
    schedule.every().minutes.do(daily_tasks) #9 AM
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    run()