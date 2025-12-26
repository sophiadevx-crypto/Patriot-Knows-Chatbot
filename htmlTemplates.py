css = '''
<style>
/* Load Noto Sans JP from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP&display=swap');

/* Apply Noto Sans JP to all elements globally — no fallback */
html, body, [class*="css"], * {
    font-family: 'Noto Sans JP' !important;
}

/* Chatbot specific styles */
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
}

.chat-message.user {
    background-color: #2b313e;
}

.chat-message.bot {
    background-color: #475063;
}

/* Avatar */
.chat-message .avatar {
    width: 20%;
}

.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}

/* Chat bubble text */
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}

/* Center the button in the sidebar */
section[data-testid="stSidebar"] .stButton > button {
    display: block;
    margin-left: auto;
    margin-right: auto;
    border-radius: 25px;
    padding: 10px 18px;
}

</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.pinimg.com/originals/f9/b5/0c/f9b50cb1406d3f4d149b16568090d88d.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.iconpacks.net/icons/2/free-user-icon-3297-thumb.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

