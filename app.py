import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import io
import os

# ReportLab Imports for PDF report generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from datetime import datetime

# ---------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon=":speech_balloon:",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------
# CUSTOM CSS (Optional Styling)
# ---------------------------
custom_css = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1f1f1f, #333333);
    color: #FAFAFA;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: #1f1f1f;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------
# LANDING PAGE / HERO SECTION
# ---------------------------
st.title("WhatsApp Chat Analyzer")
st.markdown("**Unlocking your chat insights with complete privacy!** :lock:")
st.markdown("*Created by **Mahesh & Shahid***")

# ---------------------------
# SECURITY & USAGE INSTRUCTIONS
# ---------------------------
with st.expander("Important: Security & Usage Instructions"):
    st.markdown("""
**1. Your Data Stays Local**  
We do not store or upload your chat data to any external server. All analysis is performed locally within this application session.

**2. Supported File Format**  
Please upload your WhatsApp chat export as a .txt file.  
*(Tip: In WhatsApp, use "Export Chat" > "Without Media".)*

**3. Confidentiality**  
This tool is for personal use. We recommend **not** sharing sensitive information or uploading third-party chats without consent.

**4. Data Retention**  
Once you close the browser tab or restart the app, any uploaded data is cleared from memory.

**5. Disclaimer**  
By using this tool, you agree that the analysis is provided for informational purposes only.
""")

# ---------------------------
# FILE UPLOADER SECTION
# ---------------------------
uploaded_file = st.file_uploader("Upload your exported WhatsApp chat (.txt)", type=["txt"])
if uploaded_file is not None:
    st.success(
        "File uploaded successfully! \n"
        "Now select a user (or Overall) in the left sidebar and click **Show Analysis**."
    )
else:
    st.info("Please upload a .txt file to begin analysis.")

st.markdown("---")
st.write("Made with Streamlit. © 2025 | [Privacy Policy](#) | [Terms of Service](#)")

# ---------------------------
# HELPER FUNCTION: Wrap text for PDF chart titles
# ---------------------------
def wrap_text(text, max_chars=30):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + word) <= max_chars:
            current_line += word + " "
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())
    return lines

# ---------------------------
# FUNCTION: PDF Report Generation
# ---------------------------
def generate_pdf_charts_report(charts, summary_text=""):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter
    margin = 0.5 * inch

    # --- COVER PAGE ---
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.rect(0, 0, page_width, page_height, fill=1)
    c.setLineWidth(2)
    c.setStrokeColorRGB(0, 0, 0)
    c.rect(margin, margin, page_width - 2*margin, page_height - 2*margin)
    
    # Title and other details for Cover Page
    title_y = page_height * 0.65
    tagline_y = page_height * 0.55
    summary_y = page_height * 0.45
    date_y = page_height * 0.35

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(page_width / 2, title_y, "WhatsApp Chat Analyzer Report")
    
    c.setFont("Helvetica", 14)
    c.drawCentredString(page_width / 2, tagline_y, "Unlocking your chat insights")
    
    c.setFont("Helvetica", 12)
    c.drawCentredString(page_width / 2, summary_y, summary_text)
    
    current_date = datetime.now().strftime("%B %d, %Y")
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(page_width / 2, date_y, f"Report generated on {current_date}")
    c.showPage()

    # --- CHART PAGES ---
    cols = 2
    rows = 3
    cell_width = (page_width - 2 * margin) / cols
    cell_height = (page_height - 2 * margin) / rows
    total_charts = len(charts)
    
    for i in range(0, total_charts, 6):
        c.setLineWidth(2)
        c.rect(margin, margin, page_width - 2*margin, page_height - 2*margin)
        batch = charts[i:i+6]
        for j, (chart_title, img_buffer) in enumerate(batch):
            row = j // cols
            col = j % cols
            cell_x = margin + col * cell_width
            cell_y = margin + (rows - row - 1) * cell_height

            # Wrap the title text to keep it neat within each chart cell
            lines = wrap_text(chart_title, max_chars=30)
            c.setFont("Helvetica-Bold", 12)
            line_y = cell_y + cell_height - 15
            for line in lines:
                c.drawCentredString(cell_x + cell_width/2, line_y, line)
                line_y -= 14
            title_height = 14 * len(lines) + 5

            # Add chart image into the cell area
            img_x = cell_x + 5
            img_y = cell_y + 5
            img_w = cell_width - 10
            img_h = cell_height - title_height - 10
            image = ImageReader(img_buffer)
            c.drawImage(image, img_x, img_y, width=img_w, height=img_h)

        # Last page thank you note
        if i + len(batch) >= total_charts:
            c.setFont("Helvetica-Bold", 12)
            c.drawCentredString(page_width / 2, margin + 10, "Thank you for using WhatsApp Chat Analyzer!")
            
        c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ---------------------------
# ACTUAL ANALYSIS SECTION
# ---------------------------
if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        with st.spinner("Processing chat data..."):
            df = preprocessor.preprocess(data)
    except UnicodeDecodeError:
        st.error("Error: Unable to decode the uploaded file. Please upload a valid text file.")

    if df.empty:
        st.error("Error: No messages found in the uploaded file.")
    else:
        # Convert date column to datetime and extract time info
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['time'] = df['date'].dt.strftime('%I:%M %p')
        else:
            st.error("Error: 'date' column not found in the DataFrame.")

        # Create a sorted user list with 'Overall' option at the top
        user_list = df['user'].unique().tolist()
        user_list.sort()
        user_list.insert(0, "Overall")
        selected_user = st.sidebar.selectbox("Show analysis based on users", user_list)

        if st.sidebar.button("Show Analysis"):

            # ==========================
            # CHAT AT A GLANCE (TOP STATISTICS)
            # ==========================
            st.title("Top Statistics")
            st.subheader("(Chat at a Glance)")
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.header("Total Messages")
                st.title(num_messages)
            with col2:
                st.header("Total Words")
                st.title(words)
            with col3:
                st.header("Media Shared")
                st.title(num_media_messages)
            with col4:
                st.header("Links Shared")
                st.title(num_links)

            # ==========================
            # MONTHLY TIMELINE
            # ==========================
            st.title("Monthly Timeline")
            st.subheader("(Monthly Moments)")
            timeline = helper.monthly_timeline(selected_user, df)
            try:
                timeline['time'] = pd.to_datetime(timeline['time'], errors='coerce')
            except Exception as e:
                st.error("Error converting timeline time: " + str(e))
            fig_monthly, ax = plt.subplots()
            sns.lineplot(x="time", y="message", data=timeline, marker="o", color="red", ax=ax)
            ax.set_title("Monthly Timeline (Monthly Moments)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Month", fontsize=12)
            ax.set_ylabel("Number of Messages", fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_monthly)

            # ==========================
            # DAILY TIMELINE
            # ==========================
            st.title("Daily Timeline")
            st.subheader("(Daily Dialogues)")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig_daily, ax = plt.subplots()
            sns.lineplot(x="only_date", y="message", data=daily_timeline, marker="o", color="blue", ax=ax)
            ax.set_title("Daily Timeline (Daily Dialogues)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Number of Messages", fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_daily)

            # ==========================
            # ACTIVITY MAP: Peak Activity Days & Months
            # ==========================
            st.title("Activity Map")
            st.subheader("(Peak Activity Zones)")
            col1, col2 = st.columns(2)
            with col1:
                st.header("Most Busy Day")
                busy_day = helper.week_activity_map(selected_user, df)
                fig_busy_day, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='green')
                plt.xticks(rotation='vertical')
                ax.set_title("Most Busy Day", fontsize=12, fontweight='bold')
                st.pyplot(fig_busy_day)
            with col2:
                st.header("Most Busy Month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig_busy_month, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='orange')
                plt.xticks(rotation='vertical')
                ax.set_title("Most Busy Month", fontsize=12, fontweight='bold')
                st.pyplot(fig_busy_month)

            # ==========================
            # WEEKLY ACTIVITY HEATMAP
            # ==========================
            st.title("Weekly Activity Map")
            st.subheader("(Weekly Heatmap)")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            fig_heatmap, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap, annot=True, fmt=".0f", cmap="YlGnBu")
            ax.set_title("Weekly Activity Heatmap", fontsize=12, fontweight='bold')
            st.pyplot(fig_heatmap)

            # ==========================
            # MOST BUSY USERS (Only for Overall Analysis)
            # ==========================
            if selected_user == 'Overall':
                st.title("Most Busy Users")
                st.subheader("(Top Chatter)")
                x, new_df = helper.most_busy_users(df)
                fig_most_busy, ax = plt.subplots()
                col1, col2 = st.columns(2)
                with col1:
                    ax.bar(x.index, x.values, color='red')
                    plt.xticks(rotation='vertical')
                    ax.set_title("Most Busy Users", fontsize=12, fontweight='bold')
                    st.pyplot(fig_most_busy)
                with col2:
                    st.dataframe(new_df, use_container_width=True)

            # ==========================
            # WORD CLOUD
            # ==========================
            st.title("Word Cloud")
            st.subheader("(Word Wonderland)")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig_wordcloud, ax = plt.subplots()
            ax.imshow(df_wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wordcloud)

            # ==========================
            # MOST COMMON WORDS
            # ==========================
            st.title("Most Common Words")
            st.subheader("(Key Conversations)")
            most_common_df = helper.most_common_words(selected_user, df)
            fig_common_words, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1], color='teal')
            plt.xticks(rotation='vertical')
            ax.set_title("Most Common Words", fontsize=12, fontweight='bold')
            st.pyplot(fig_common_words)

            # ==========================
            # ENGAGEMENT & RESPONSE TIME ANALYSIS
            # ==========================
            st.title("Engagement & Response Time Analysis")
            st.subheader("(Engagement Breakdown)")
            response_df = helper.response_time_analysis(selected_user, df)
            st.write("**Average Response Time (in hours) per User:**")
            st.dataframe(response_df, use_container_width=True)

            # ---------------------------
            # SILENT OBSERVERS (Only for Overall Analysis)
            # ---------------------------
            if selected_user == 'Overall':
                st.title("Silent Observers List")
                st.subheader("(Hidden Listeners)")
                silent_list_df = helper.silent_observers(df)
                st.write("**Users with the lowest message counts:**")
                st.dataframe(silent_list_df, use_container_width=True)

            # ==========================
            # EMOJI ANALYSIS
            # ==========================
            st.title("Emoji Analysis")
            st.subheader("(Emoji Insights)")
            emoji_df = helper.emoji_helper(selected_user, df)
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df, use_container_width=True)
            with col2:
                fig_emoji = None
                if not emoji_df.empty:
                    fig_emoji, ax = plt.subplots()
                    num = min(5, len(emoji_df))
                    ax.pie(emoji_df["count"].head(num), labels=emoji_df["emoji"].head(num), autopct="%0.2f")
                    plt.setp(ax.texts, fontname="Segoe UI Emoji")
                    ax.set_title("Top Emojis", fontsize=12, fontweight='bold')
                    st.pyplot(fig_emoji)
                else:
                    st.warning("No emojis found in the chat.")

            # ==========================
            # CONVERSATION NETWORK GRAPH (Only for Overall Analysis)
            # ==========================
            if selected_user == 'Overall':
                st.title("Conversation Network Graph")
                st.subheader("(Chat Connections)")
                graph = helper.conversation_network_graph(selected_user, df)
                fig_network, ax = plt.subplots(figsize=(8, 6))
                pos = nx.spring_layout(graph, k=0.5)
                nx.draw_networkx_nodes(graph, pos, node_color='orange', node_size=500, ax=ax)
                nx.draw_networkx_edges(graph, pos, edge_color='lightblue', width=2, ax=ax)
                nx.draw_networkx_labels(graph, pos, font_color='black', ax=ax)
                ax.set_title("Conversation Network Graph", fontsize=12, fontweight='bold')
                plt.axis("off")
                st.pyplot(fig_network)

            # ==========================
            # TOPIC MODELING & KEYWORD EXTRACTION
            # ==========================
            st.title("Topic Modeling & Keyword Extraction")
            st.subheader("(Topic Trends)")
            topics = helper.topic_modeling(selected_user, df, num_topics=3)
            st.write("**Discovered Topics:**")
            st.write(topics)

            # ==========================
            # SENTIMENT ANALYSIS
            # ==========================
            st.title("Sentiment Analysis")
            st.subheader("(Mood Overview)")
            sentiment_df = helper.sentiment_analysis(selected_user, df)
            st.write("**Sentiment Counts and Percentages:**")
            st.dataframe(sentiment_df, use_container_width=True)
            fig_sentiment, ax = plt.subplots()
            ax.pie(
                sentiment_df['Count'],
                labels=sentiment_df['Sentiment'],
                autopct='%1.1f%%',
                startangle=140
            )
            ax.axis('equal')
            ax.set_title("Sentiment Analysis", fontsize=12, fontweight='bold')
            st.pyplot(fig_sentiment)

            st.title("Thanks for using WhatsApp Chat Analyzer!")

            # ==========================
            # PREPARE SUMMARY FOR COVER PAGE
            # ==========================
            summary_text = f"Total Messages: {num_messages} | Total Words: {words} | Media Shared: {num_media_messages} | Links Shared: {num_links}"

            # ==========================
            # CAPTURE CHARTS AS IMAGES FOR PDF REPORT
            # ==========================
            charts_list = []
            buf = io.BytesIO()
            fig_monthly.savefig(buf, format="PNG")
            buf.seek(0)
            charts_list.append(("Monthly Timeline (Monthly Moments)", buf))

            buf = io.BytesIO()
            fig_daily.savefig(buf, format="PNG")
            buf.seek(0)
            charts_list.append(("Daily Timeline (Daily Dialogues)", buf))

            buf = io.BytesIO()
            fig_busy_day.savefig(buf, format="PNG")
            buf.seek(0)
            charts_list.append(("Most Busy Day (Peak Activity Zones)", buf))

            buf = io.BytesIO()
            fig_busy_month.savefig(buf, format="PNG")
            buf.seek(0)
            charts_list.append(("Most Busy Month (Peak Activity Zones)", buf))

            buf = io.BytesIO()
            fig_heatmap.savefig(buf, format="PNG")
            buf.seek(0)
            charts_list.append(("Weekly Activity Heatmap (Weekly Heatmap)", buf))

            buf = io.BytesIO()
            fig_wordcloud.savefig(buf, format="PNG")
            buf.seek(0)
            charts_list.append(("Word Cloud (Word Wonderland)", buf))

            buf = io.BytesIO()
            fig_common_words.savefig(buf, format="PNG")
            buf.seek(0)
            charts_list.append(("Most Common Words (Key Conversations)", buf))

            if fig_emoji is not None:
                buf = io.BytesIO()
                fig_emoji.savefig(buf, format="PNG")
                buf.seek(0)
                charts_list.append(("Emoji Analysis (Emoji Insights)", buf))

            if selected_user == 'Overall':
                buf = io.BytesIO()
                fig_network.savefig(buf, format="PNG")
                buf.seek(0)
                charts_list.append(("Conversation Network Graph (Chat Connections)", buf))

            buf = io.BytesIO()
            fig_sentiment.savefig(buf, format="PNG")
            buf.seek(0)
            charts_list.append(("Sentiment Analysis (Mood Overview)", buf))

            # Generate and provide PDF download button
            pdf_buffer = generate_pdf_charts_report(charts_list, summary_text=summary_text)
            st.download_button(
                label="Download PDF - Get Your Visual Chat Report!",
                data=pdf_buffer,
                file_name="chat_analysis_visual_report.pdf",
                mime="application/pdf"
            )

            # ==========================
            # CONTEXTUAL INSIGHTS & RECOMMENDATIONS
            # ==========================
            st.title("Contextual Insights and Recommendations")

            with st.expander("Click here to reveal insights"):
                insights = []

                # Insight 1: Identify the Peak Activity Day and Hour
                if not df.empty:
                    # Peak Day by message count
                    daily_counts = df.groupby(df['date'].dt.date).size()
                    if not daily_counts.empty:
                        peak_day = daily_counts.idxmax()
                        peak_messages = daily_counts.max()
                        insights.append(f"📅 **Peak Activity Day:** {peak_day.strftime('%A, %d %B %Y')} with {peak_messages} messages.")
                    
                    # Peak Hour of the day by message count
                    hourly_counts = df['hour'].value_counts()
                    if not hourly_counts.empty:
                        peak_hour = hourly_counts.idxmax()
                        insights.append(f"⏰ **Peak Hour:** Around {peak_hour}:00 hrs with maximum activity.")

                # Insight 2: Dominant Sentiment
                sent_df = helper.sentiment_analysis(selected_user, df)
                if not sent_df.empty:
                    dominant_sent = sent_df.loc[sent_df['Count'].idxmax()]['Sentiment']
                    insights.append(f"😊 **Dominant Sentiment:** {dominant_sent}.")
                
                # Insight 3: Engagement Patterns
                if not response_df.empty:
                    avg_response = response_df['response_time'].mean()
                    insights.append(f"⏱ **Average Response Time:** {round(avg_response, 2)} hours (lower is generally better for engagement).")
                
                # Insight 4: Emoji Vibes (if available)
                if not emoji_df.empty:
                    top_emoji = emoji_df.iloc[0]
                    insights.append(f"😁 **Top Emoji:** {top_emoji['emoji']} used {top_emoji['count']} times, indicating a fun conversation!")

                # Combine insights and add recommendations based on observed patterns
                recommendations = []
                
                # Recommendation based on overall activity level
                if num_messages < 100:
                    recommendations.append("It seems there is not much activity. Consider initiating more discussions to engage everyone!")
                elif num_messages > 1000:
                    recommendations.append("High activity detected! You might want to highlight important messages or create sub-groups for focused discussions.")
                else:
                    recommendations.append("The activity level looks balanced. Keep up the engaging conversations!")
                
                # Recommendation based on response time
                if not response_df.empty and response_df['response_time'].mean() > 1:
                    recommendations.append("The average response time is a bit high. Encourage quicker responses to maintain the conversation flow.")
                else:
                    recommendations.append("Response times are prompt, which is a good sign of active engagement!")
                
                # Recommendation based on sentiment
                if not sent_df.empty and dominant_sent.lower() in ["negative", "sad"]:
                    recommendations.append("The dominant sentiment is negative. It might help to inject some positive topics or humor into the chat.")
                else:
                    recommendations.append("The overall sentiment appears positive. Keep fostering that upbeat atmosphere!")
                
                # Display all insights
                st.markdown("### Key Insights:")
                for insight in insights:
                    st.markdown(f"- {insight}")
                
                # Display recommendations
                st.markdown("### Recommendations:")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
