import re
import pandas as pd

def preprocess(data):
    """
    Preprocesses raw WhatsApp chat data and returns a formatted DataFrame.

    This function performs the following tasks:
      1. Splits the raw chat data using a regular expression pattern that matches the 12-hour date-time format.
      2. Extracts messages and their corresponding timestamps.
      3. Converts timestamps to datetime objects.
      4. Separates the sender (user) from the message content.
      5. Creates additional time-related columns (date, month, day, hour, etc.) for further analysis.

    Parameters:
        data (str): The raw chat data as a single string.

    Returns:
        DataFrame: A pandas DataFrame containing the processed chat data with columns:
                   - date: Message timestamp as a datetime object.
                   - user: Username or "group_notification" for system messages.
                   - message: The actual text content of the message.
                   - only_date: Date part of the timestamp (without time).
                   - year, month_num, month, day, day_name, hour, minute: Time-related fields.
                   - period: A string representing the time period (e.g., "2 PM - 3 PM").
                   - time: Formatted time in 12-hour format with AM/PM.
    """
    # Define a regex pattern to match the date and time stamp (e.g., "12/31/20, 9:35 PM - ")
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][Mm]\s-\s'
    
    # Split the raw data into messages using the pattern and extract the matching date-time stamps
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    
    # Create a DataFrame with the extracted messages and their corresponding date-time stamps
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    
    # Convert the date-time strings into pandas datetime objects using the provided format
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p - ')
    
    # Rename the column for clarity
    df.rename(columns={'message_date': 'date'}, inplace=True)
    
    # Extract the user and message text from the 'user_message' column
    users = []
    messages_list = []
    for message in df['user_message']:
        # Split the message using a pattern that captures the sender's name before ": "
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages_list.append(" ".join(entry[2:]))
        else:
            # If no sender is found, mark as 'group_notification'
            users.append('group_notification')
            messages_list.append(entry[0])
    
    # Create separate 'user' and 'message' columns and drop the temporary 'user_message' column
    df['user'] = users
    df['message'] = messages_list
    df.drop(columns=['user_message'], inplace=True)
    
    # Create additional columns for easier time-based analysis
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Create a "period" column that groups time into intervals (e.g., "1 PM - 2 PM")
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append("11 PM - 12 AM")
        elif hour == 0:
            period.append("12 AM - 1 AM")
        elif hour < 12:
            period.append(f"{hour} AM - {hour+1} AM")
        elif hour == 12:
            period.append("12 PM - 1 PM")
        else:
            period.append(f"{hour-12} PM - {hour-11} PM")
    df['period'] = period

    # Format the 'time' column in a 12-hour format with AM/PM
    df['time'] = df['date'].dt.strftime('%I:%M %p')
    
    return df
