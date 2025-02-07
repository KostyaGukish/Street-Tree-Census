FROM python:3.12-slim

COPY . .

RUN pip install -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "street_tree_census.server:app", "--reload", "--host", "0.0.0.0", "--port", "80"]