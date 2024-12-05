import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                
from notification.mail_sender import MailSender

sender = MailSender()

sender.send_mail("Test Email from Python", "This is a test email sent using Python and Libero Mail.")
