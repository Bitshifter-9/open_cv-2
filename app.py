import streamlit as st
import pandas as pd 
import time

from datetime import datetime
ts=time.time()
date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")

import os 
from sqlalchemy import create_engine

# Database connection
try:
    if 'DATABASE_URL' in os.environ:
        DATABASE_URL = os.environ['DATABASE_URL']
        engine = create_engine(DATABASE_URL)
        
        query = f"SELECT * FROM attendance WHERE \"Date\" = '{date}'"
        df = pd.read_sql(query, engine)
        
        st.dataframe(df.style.highlight_max(axis=0))
    else:
        raise Exception("DATABASE_URL not set")

except Exception as e:
    # Fallback to CSV
    try:
        df = pd.read_csv("attendance/Attendance_"+date+".csv")
        st.dataframe(df.style.highlight_max(axis=0))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.write(f"No attendance records found for {date}")