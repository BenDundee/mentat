# Front End Specification

## Front End: High Level Outline
Mentat is an executive coach, the primary interface is a chatbot. The functionality should be largely
the same as the front end of Claude or Gemini:
  - **General UX** The user is presented with a new chat window when they log in, along with the option to continue a 
previous session. If a previous session is loaded, the conversation state on the back end should be 
persisted, so that the user may pick up where they left off.
  - **Document Upload** The user may attach documents to the message, and upload them to the system. For now, the user 
is not expected to have access to uploaded documents, once they've been uploaded they will be persisted in the
document store. In the future we *may* decide to have a lightweight file system and file viewer.
  - **Enter Button** This has always bugged me about chatbot apps: The enter button should be a carriage 
return, and Shift+enter should submit the message to the back end. I am frustrated by the design of chatbot
apps today, so I have opinions! To be more logical: The conversations with Mentat are expected to be longer,
and I expect users (read: me) to answer with more than one paragraph.

## Logging in the Front End
Since the back end will take a while to think, I would like to surface some logging messages to the user
in the chat window. You (Claude Code) do this while you're thinking (I've seen "Contemplating", "Swirling" 
and "Tomfoolering", for example). The user-facing logging messages should be informative, and descriptive of
what the back end is doing. They are also ephemeral, in the sense that they are not persisted once the 
coach returns its final answer.

## Conversation History and Conversation Page
Each previous conversation should appear in the left rail, similar to familiar chatbot interfaces. Clicking on 
the conversation will load the history into the chat window, and the user may continue the conversation.

Each previous conversation will also have a link to a "Conversation Page". In the future, there will be an 
AI-generated summary with action items and takeaways, and users will have the ability to revisit or continue 
the conversation. For now, the last few messages will appear on the Conversation Page, it is ok for it to 
be mostly blank.

## Authentication and User Management
I want a simple authentication framework for personal use and testing. In the future we will use Google OAuth, 
so let's make our MVP easy to replace with something production-grade.

### User Profile Page
For now, the user profile page should be simple. In the future, the profile page will contain an AI-generated bio of 
the user, that is updated as the system learns more.

In the future it will contain a links to the following pages:
  - The Coaching Plan Page, a page with current goals, assignments, and action items. The user will have an option to 
discuss the Coaching Plan with the Coach at any time (ie, "Discuss this plan" button that opens a chat with a specific
prompt).
  - The Assignments Page, with any open goals, assignments, and action items, along with due dates. Users will have an 
opportunity to discuss any of these with the Coach at any time (ie, "Discuss my assignments" button that opens a chat
window with a specific prompt).
  - The Assessment Page. This page will contain an up-to-date assessment of the user, their strengths, weaknesses, and 
any observations the Coach has made about opportunities.
  - The Career Management Page. This page will contain the User's career history in detail. In the future, users will 
have the ability to initiate a conversation about a job opportunity (ie, "Discuss a job opportunity" button that opens 
a chat window with a specific prompt). The Coach will help the User discuss the role they're considering, how their 
strengths, weaknesses, and goals align. The Coach will also help by generating a resume that positions the candidate for 
success.