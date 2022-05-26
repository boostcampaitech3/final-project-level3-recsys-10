run_server:
	python3 -m backend

run_client:
	python3 -m streamlit run frontend/ui.py --server.port 30003

run_app : run_server run_client