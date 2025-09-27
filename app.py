import json
import os
import time
import uuid
from datetime import datetime, timedelta

import streamlit as st
import supabase
from dotenv import load_dotenv
from src.core.rag_chain import AdvancedRAGChain

# Load environment variables
load_dotenv()

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)


def initialize_session_state():
    """Initialize Streamlit session state with user authentication and chat history."""
    if "user" not in st.session_state:
        st.session_state.user = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = AdvancedRAGChain()


def authenticate_user():
    """Display a login form and handle authentication."""
    st.title("🔐 Login to Rocky")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        try:
            print(f"Attempting login with email: {email}")
            # Store the full response, not just the user
            response = supabase_client.auth.sign_in_with_password({"email": email, "password": password})

            # Store both user and session data
            if response.user:
                st.session_state.user = response.user
                st.session_state.session = response.session  # Store the session object
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    if st.button("Create Account"):
        try:
            response = supabase_client.auth.sign_up({"email": email, "password": password})

            user = response.user if hasattr(response, 'user') else response.get('user')

            if user:
                # Automatically log the user in after account creation
                st.session_state.user = user
                st.session_state.session = response.session  # Store the session object
                st.success("✅ Account created! Please login.")
                print(f"Account created: {user.email if hasattr(user, 'email') else user.get('email')}")
                st.rerun()
            else:
                st.error("❌ Could not create account")
                print("No user in signup response")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")


# At the beginning of your application
def get_supabase_client():
    """Get Supabase client with auth token if available"""
    client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

    if "session" in st.session_state and st.session_state.session:
        try:
            # Set the auth session on the client
            client.auth.set_session(
                st.session_state.session.access_token,
                st.session_state.session.refresh_token
            )
        except Exception as e:
            print(f"Error setting auth session: {e}")

    return client


def load_conversations(user_id):
    """Load all past conversations of the user."""
    client = get_supabase_client()
    response = client.table("chat_history").select("conversation_id, created_at").eq("user_id", user_id).order(
        "created_at", desc=True).execute()

    if response.data:
        return response.data
    return []


def load_user_history(user_id, conversation_id):
    """Load a specific conversation history for a user from Supabase."""
    client = get_supabase_client()
    response = client.table("chat_history").select("messages").eq("user_id", user_id).eq("conversation_id",
                                                                                         conversation_id).execute()

    if response.data:
        return json.loads(response.data[0]["messages"])  # Return the messages list
    return []  # Return an empty list if no conversation is found


def save_user_history(user_id, conversation_id):
    """Save conversation history in Supabase."""
    client = get_supabase_client()
    data = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "messages": json.dumps(st.session_state.messages)
    }

    # Check if the conversation exists
    response = client.table("chat_history").select("id").eq("user_id", user_id).eq("conversation_id",
                                                                                   conversation_id).execute()

    if response.data:
        # Update existing conversation
        client.table("chat_history").update(data).eq("conversation_id", conversation_id).eq("user_id",
                                                                                            user_id).execute()
    else:
        # Insert new conversation
        client.table("chat_history").insert(data).execute()


def chat_interface():
    """Main chatbot interface with a sidebar for conversation history."""

    # Sidebar for past conversations
    st.sidebar.title("🗂 Past Conversations")

    if st.session_state.user:
        # Get user ID safely
        try:
            if hasattr(st.session_state.user, "id"):
                user_id = st.session_state.user.id
            elif isinstance(st.session_state.user, dict) and "id" in st.session_state.user:
                user_id = st.session_state.user["id"]
            else:
                st.error("Unable to find user ID")
                return
        except Exception as e:
            st.error(f"Error accessing user ID: {e}")
            return

        # Load past conversations
        conversations = load_conversations(user_id)

        # Button to start a new conversation
        if st.sidebar.button("➕ New Conversation"):
            st.session_state.messages = []  # Reset messages
            # Always generate a new ID when creating a new conversation
            st.session_state.current_conversation_id = str(uuid.uuid4())
            # Save the new empty conversation
            save_user_history(user_id, st.session_state.current_conversation_id)
            st.rerun()

        # Display past conversations in the sidebar
        if conversations:
            for conv in conversations:

                # Parse the timestamp string
                timestamp = datetime.fromisoformat(conv['created_at'].replace('Z', '+00:00'))
                # Add one hour to adjust for your timezone
                local_time = timestamp + timedelta(hours=1)
                # Format it nicely
                formatted_time = local_time.strftime("%d/%m/%Y %H:%M")

                button_text = f"🗨️ {formatted_time}"

                if st.sidebar.button(button_text, key=f"conv_{conv['conversation_id']}"):
                    st.session_state.messages = load_user_history(user_id, conv["conversation_id"])
                    st.session_state.current_conversation_id = conv["conversation_id"]
                    st.rerun()
        else:
            st.sidebar.write("No past conversations.")

    # Main layout
    col1, col2 = st.columns([5, 1])

    with col1:
        st.title("Rocky - ERT's AI Assistant")

    with col2:
        if st.session_state.user and st.button("Logout", key="logout_top"):
            st.session_state.user = None
            st.rerun()

    st.caption("Your knowledgeable companion for rocket engineering 🚀")

    if st.session_state.user:
        # Display chat history messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Handle user input
        # In chat_interface() function, update the user input handler:
        if prompt := st.chat_input("How can I help you today?"):
            # Make sure we have a conversation_id before adding messages
            if "current_conversation_id" not in st.session_state or st.session_state.current_conversation_id is None:
                st.session_state.current_conversation_id = str(uuid.uuid4())

            # Now use the established conversation_id
            conversation_id = st.session_state.current_conversation_id
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                with st.spinner("Rocky is thinking..."):
                    full_response = ""

                    chat_history = []
                    messages = st.session_state.messages
                    for i in range(0, len(messages) - 1, 2):
                        chat_history.append((messages[i]["content"], messages[i + 1]["content"]))

                    for chunk in st.session_state.rag_chain.stream_response(prompt, chat_history):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                        time.sleep(0.04)

                    placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_user_history(user_id, conversation_id)

    else:
        authenticate_user()  # Show login page if a user is not logged in


def main():
    """Main function to handle authentication and chat interface."""
    initialize_session_state()

    if st.session_state.user:
        chat_interface()
    else:
        authenticate_user()


if __name__ == "__main__":
    main()
