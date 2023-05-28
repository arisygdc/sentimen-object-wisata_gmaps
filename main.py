from st_pages import show_pages_from_config
import os
show_pages_from_config()

# check if .hasil folder exists
if not os.path.exists('.hasil'):
    os.makedirs('.hasil')
