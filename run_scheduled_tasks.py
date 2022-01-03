import schedule
import time
from utils.send_email import send_email
 
def daily_tasks():
    #TODO run daily tasks
    send_email(["1"])
  
schedule.every().day.at("09:00").do(daily_tasks) #9 AM

def run():
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    run()