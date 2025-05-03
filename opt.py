import smtplib


# Body content of the email
def body_scam():
    return "Dear user,\nWe have received your feedback. We will get back to you soon.\nRegards,\nAtScanner Support team"

# Wrapper (decorator) function to handle email sending
def deaf(func):
    def internal(*args, **kwargs):
        # Email settings
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()        
        password = "bqll dtoe ctsz skwb"  # App-specific password
        server.login("atscanner05@gmail.com", password)
        # Call the original function and get the message to send
        result = func(*args, **kwargs)
        
        # Email details
        receiver_email = kwargs.get('receiver_email')  # Receiver's email must be passed as a keyword argument
        if receiver_email is None:
            raise ValueError("receiver_email must be provided")

        # Send the email
        server.sendmail("atscanner05@gmail.com", receiver_email, result)  # Use result as the email message
        server.quit()

        return result
    return internal

# Function to be wrapped
@deaf
def send_feedback_mail(receiver_email):
    # Define email subject and body
    subject = "Thank you for your feedback!"
    body = body_scam()
    message = f'Subject: {subject}\n\n{body}'
    
    return message  # Return the message to be sent by the decorator

# Example usage
