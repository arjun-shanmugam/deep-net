import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "nba.preds.by.the.big.dawgs@gmail.com"  # Enter your address
receiver_emails = ["andrew_delworth@brown.edu"]  # Enter receiver address
password = "bigdawgsgottaeat"
message = """\
Subject: Hi there

This message is sent from Python."""

for receiver_email in receiver_emails:
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)