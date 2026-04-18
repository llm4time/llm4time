import streamlit as st
from streamlit_theme import st_theme
import storage.repository as repo
import storage.exceptions as exc
from PIL import Image
import io

auth = st.session_state.auth


# =============================================================================
# Theme & Styles
# =============================================================================

theme = st_theme()
if theme is None:
  st.stop()

st.html(f"""
<style>
.st-key-profile_picture_popover {{
  position: relative;
  overflow: hidden;
  padding: 0;
}}
.stPopover button {{
  width: 156px !important;
  height: 156px !important;
  position: absolute;
  z-index: 999;
  opacity: 0;
  top: 0;
}}
.st-key-profile_picture {{
  position: absolute;
  z-index: 999;
  opacity: 0;
  top: 0;
}}
.st-key-profile_picture:hover + .st-key-upload_button button {{
  background: {theme["darkenedBgMix15"]};
}}
.st-key-profile_picture,
.st-key-profile_picture * {{
  height: 2.5rem !important;
}}
.st-key-profile_picture section {{
  padding: 0;
}}
.st-key-profile_picture span,
.st-key-profile_picture label,
.st-key-profile_picture .stFileUploader > div {{
  display: none;
}}
</style>
""")


# =============================================================================
# Image Processing
# =============================================================================

def crop_center_square(img: Image.Image) -> Image.Image:
  w, h = img.size
  m = min(w, h)
  left = (w - m) // 2
  top = (h - m) // 2
  return img.crop((left, top, left + m, top + m))


def update_profile_picture(picture: bytes | None) -> None:
  try:
    repo.users().update_picture(user_id=auth.user.sub, picture=picture)
    return True
  except Exception as e:
    print("Error updating profile picture:", e)
    st.toast("Failed to update profile picture", icon="🚨")
    return False


# =============================================================================
# UI Components - Upload Profile Picture Dialog
# =============================================================================

@st.dialog("Preview Profile Picture")
def upload_profile_picture_dialog() -> None:
  uploaded = st.session_state.profile_picture

  with st.container(horizontal_alignment="center"):
    raw_bytes = uploaded.getvalue()
    cropped_img = Image.open(io.BytesIO(raw_bytes))
    cropped_img = crop_center_square(cropped_img)

    st.image(cropped_img, width=256, output_format="PNG")

    if st.button("Set new profile picture", type="primary", width="stretch"):
      buf = io.BytesIO()
      cropped_img.save(buf, format="PNG")
      cropped_bytes = buf.getvalue()
      if update_profile_picture(cropped_bytes):
        auth.user.picture = cropped_bytes
        st.rerun()


# =============================================================================
# UI Components - Delete Account Dialog
# =============================================================================


@st.dialog("Are you ABSOLUTELY sure?")
def delete_account_dialog() -> None:
  st.write(
      f"This will permanently delete the **{auth.user.username}** account, datasets, models, prompts, and history."
  )

  st.caption("**⚠️ This action CANNOT be undone.**")

  username = st.text_input("Please enter your username to confirm.")
  username_mismatch = username != auth.user.username

  if st.button("**I understand the consequences, delete this account**", type="primary", width="stretch", disabled=username_mismatch):
    auth.delete_account()


# =============================================================================
# UI Components - User Info
# =============================================================================

with st.container(horizontal_alignment="center"):

  with st.container(width=156, height=156, key="profile_picture_popover"):
    st.image(auth.user.picture, width="stretch")
    popover = st.popover("Filter items", width="stretch", help="Change profile picture")

    popover.file_uploader(
        "Upload profile picture",
        type=["png", "jpg", "jpeg"],
        key="profile_picture",
        on_change=upload_profile_picture_dialog,
        width="stretch"
    )
    popover.button("Upload", icon=":material/upload:", width="stretch",
                   key="upload_button")

    if popover.button("Remove", type="primary", icon=":material/delete:",
                      width="stretch"):
      if update_profile_picture(None):
        auth.user.picture = None
        st.rerun()

  st.text_input("ID", auth.user.sub, icon=":material/tag:", disabled=True)

  st.write("---")
  username = st.text_input("Username", auth.user.username, icon=":material/person:")
  email = st.text_input("Email", auth.user.email, icon=":material/mail:")
  if st.button("Save Changes", icon=":material/save:", width="stretch"):
    try:
      repo.users().update(user_id=auth.user.sub, username=username, email=email)
      st.toast("Account details updated", icon="✅")
    except exc.UsernameAlreadyExistsError:
      st.warning("Username already exists.")
    except exc.EmailAlreadyExistsError:
      st.warning("Email already exists.")
    except Exception as e:
      st.warning("Failed to update account details")

  st.write("---")
  st.button("Logout", width="stretch", icon=":material/logout:", on_click=auth.signout)
  st.button("Delete Account", type="primary", width="stretch",
            icon=":material/delete:", on_click=delete_account_dialog)
