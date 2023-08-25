# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:04:44 2023

@author: MaxGr
"""


import pytoml
from sqlalchemy import create_engine, text

def upload():
    # Database connection details
    user = ""
    passw = ""
    host = ""
    schema = ""
    
    with open('mysql-credentials.toml') as f:
        c = pytoml.load(f)

    # Create a database engine
    engine = create_engine(f"mysql+pymysql://{user}:{passw}@{host}/{schema}".format(**c))

    # Print the engine details
    print(f"{engine=}")
    
    query = """
    SELECT *
        FROM status
        WHERE building = '10.21'
        
            
    

    # # Establish a connection and execute queries
    # with engine.connect() as con:
    #     # Create 'status' table if it doesn't exist
    #     con.execute(text(
    #         """
    #         CREATE TABLE IF NOT EXISTS status (
    #             id INT AUTO_INCREMENT PRIMARY KEY,
    #             status_name VARCHAR(255) NOT NULL
    #         )
    #         """
    #     ))
    #     print("Table 'status' created successfully.")

    #     # Insert data into 'status' table
    #     statuses = ["Active", "Inactive"]
    #     for status in statuses:
    #         con.execute(text(
    #             """
    #             INSERT INTO status (status_name) VALUES (:status_name)
    #             """
    #         ), status_name=status)
    #     print("Data inserted successfully.")



        # Retrieve and print data from 'status' table
        result = con.execute(text(f"SELECT * FROM status"))
        for row in result:
            print(f"{row=}")






# if __name__ == "__main__":
#     main()











