import pandas as pd
import io
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders
import os

class email_send():
    def __init__(self,users_non_compliant_dict,tag):
        self.users_non_compliant_dict = users_non_compliant_dict
        self.tag = tag
        # Read SMTP configuration from environment when available
        self.SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
        self.SMTP_USERNAME = os.getenv('SMTP_USERNAME', os.getenv('MAIL_USERNAME', ''))
        self.SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', os.getenv('MAIL_PASSWORD', ''))
        self.SENDER_EMAIL = os.getenv('SENDER_EMAIL', self.SMTP_USERNAME or 'no-reply@example.com')
        self.SENDER_NAME = os.getenv('SENDER_NAME', 'KYC Ops')

        #self.attachments = ['C:/EY FSRM Tools/KYC Verify/Backend/entity_docs/for emailing/KYC_form.pdf']  # Path to your PDF file
        self.folder_path_RFI = "C:/Users/ZH168VY/OneDrive - EY/2026/Backend/entity_docs/Customer Outreach Email/RFI"
        self.folder_path_EDD = "C:/Users/ZH168VY/OneDrive - EY/2026/Backend/entity_docs/Customer Outreach Email/EDD"                   
        
    def email_send(self): 
        #For sending email with attachment
        # If SMTP is not configured, skip real sending and simulate results
        simulate_only = not (self.SMTP_USERNAME and self.SMTP_PASSWORD)
        if simulate_only:
            print("SMTP not configured — simulating email sends (no network calls)")
        df = pd.DataFrame(self.users_non_compliant_dict)
        #print(df.head())
        with smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT) as server:
            server.starttls()
            server.login(self.SMTP_USERNAME, self.SMTP_PASSWORD)

            # Send an email to each non-compliant customer
            non_compliant_customers = df[df['CompliantStatus'] == 'Non-Compliant']
            output = list()
            for index, row in non_compliant_customers.iterrows():
                msg = MIMEMultipart()
                msg['From'] = f"{self.SENDER_NAME} <{self.SENDER_EMAIL}>"
                msg['To'] = row['EmailId']

                # Customize the email body as needed
                if self.tag == "rfi":
                    msg['Subject'] = 'Request For Information'
                    folder_path = self.folder_path_RFI
                    body = f"Dear {row['CustomerID']},\n\n" \
                           f"We have identified that few documents need to be updated for KYC completion. List of the documents to be shared is attached herein. \n" \
                           f"\t Passport, Utility Bill, Ration Card, ITR.\n\n" \
                           f"Please share this document in response to this email.\n\n" \
                           f"Best Regards,\n" \
                           f"{self.SENDER_NAME}\n"\
                           f"Manager,\n" \
                           f"KYC Ops"
                elif self.tag == "edd":
                    msg['Subject'] = 'Enhanced Due Diligence'
                    folder_path = self.folder_path_EDD
                    body = "NA"

                msg.attach(MIMEText(body, 'plain'))

                # Attach files from folder if present
                if os.path.isdir(folder_path):
                    for file_name in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file_name)
                        if os.path.isfile(file_path):
                            self.attach_file(msg, file_path)
                else:
                    print(f"Error: Folder {folder_path} not found.")

                if simulate_only:
                    output.append(f"Simulated send to {row.get('EmailId')} for {row.get('CustomerID')}")
                else:
                    # perform real send
                    try:
                        with smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT) as server:
                            server.starttls()
                            server.login(self.SMTP_USERNAME, self.SMTP_PASSWORD)
                            server.send_message(msg)
                        output.append(f"Email sent successfully to {row.get('CustomerID', row.get('EmailId'))}")
                    except Exception as e:
                        err = f"Failed to send to {row.get('EmailId')}: {e}"
                        print(err)
                        output.append(err)
        return {"output" : output}


    def attach_file(self,msg, file_path):
        # Check if file exists
        if not os.path.isfile(file_path):
            print(f"Error: File {file_path} not found.")
            return

        file_name = os.path.basename(file_path)
        # Guess MIME type based on file extension
        mime_type = ('application', 'octet-stream')  # Default for unknown types
        if file_name.lower().endswith('.pdf'):
            mime_type = ('application', 'pdf')
        elif file_name.lower().endswith('.docx'):
            mime_type = ('application', 'vnd.openxmlformats-officedocument.wordprocessingml.document')

        # Read the file and create a MIMEBase object
        with open(file_path, 'rb') as attachment:
            part = MIMEBase(*mime_type)
            part.set_payload(attachment.read())

        # Encode the file as base64
        encoders.encode_base64(part)

        # Add header to the attachment
        part.add_header('Content-Disposition', f'attachment; filename= {file_name}')

        # Attach the file to the email
        msg.attach(part)



"""
    def email_sending(self): 
    #For sending email without attachment

            df = pd.DataFrame(self.users_non_compliant_dict)
            #print(df.head())
            with smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT) as server:
                server.starttls()
                server.login(self.SMTP_USERNAME, self.SMTP_PASSWORD)

                # Send an email to each non-compliant customer
                non_compliant_customers = df[df['CompliantStatus'] == 'Non-Compliant']
                output = list()
                for index, row in non_compliant_customers.iterrows():
                    msg = MIMEMultipart()
                    msg['From'] = f"{self.SENDER_NAME} <{self.SENDER_EMAIL}>"
                    msg['To'] = row['EmailId']
                    msg['Subject'] = 'Compliance Issue'

                    # Customize the email body as needed
                    body = f"Dear {row['CustomerID']},\n\n" \
                            f"We have identified the following compliance issue with your account: {row['reason']} for KYC refresh.\n\n" \
                            f"Please address this issue as soon as possible.\n\n" \
                            f"Upload required document in the given link http://localhost:3000/#/mailForm/Documents \n\n"\
                            f"Best Regards,\n\n" \
                            f"{self.SENDER_NAME}\n"\
                            f"Manager,\n" \
                            f"Bank, Mumbai" 
                    
                    msg.attach(MIMEText(body, 'plain'))
                    server.send_message(msg)
                    message = f"Email sent successfully to {row['CustomerID']}"
                    output.append(message)
                    #print(message)
            return {"output" : output}

"""
