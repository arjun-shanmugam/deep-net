import smtplib, ssl, datetime

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "nba.preds.by.the.big.dawgs@gmail.com"  # Enter your address
receiver_emails = ["andrew_delworth@brown.edu"]  # Enter receiver address
password = "bigdawgsgottaeat"
message = """\
Subject: Big Dawgs Predictions - {}

Predictions\n"""

#take list of predictions, send email to list of users
#TODO - Predictions argument needs to be well-defined, need to
#be able to identify which game each pred is for
def send_email(predictions):
    prediction_string = message
    for pred in predictions:
        prediction_string += pred + "\n"
    for receiver_email in receiver_emails:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, prediction_string.format(datetime.date.today()))