mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"118EE0326@nitrkl.ac.in\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
