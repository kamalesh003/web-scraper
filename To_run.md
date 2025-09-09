#Basic Usage
python app1.py --url https://example.com

# With custom depth and threads
python app1.py --url https://example.com --depth 2 --threads 3

# Faster scraping (lower delay, ignore robots.txt)
python app1.py --url https://example.com --delay 0.1 --no-robots

# Generate embeddings and use fallback extraction
python app1.py --url https://example.com --embeddings --no-trafilatura

# Custom output file names
python app1.py --url https://example.com --data-file my_data.json --log-file my_log.txt

# All options combined
python app1.py --url https://example.com --depth 4 --threads 8 --delay 0.2 --no-trafilatura --no-robots --embeddings



---

Eg:

>python app1.py --url "https://www.geeksforgeeks.org/machine-learning/bagging-vs-boosting-in-machine-learning/" --depth 1 --embeddings --data-file my_data.json --embeddings-file my_embeddings.json
