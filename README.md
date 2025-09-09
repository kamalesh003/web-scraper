# Web Scraper Command Line Arguments Documentation

## Basic Usage
```cmd
python app1.py --url https://example.com
```

## Argument Reference

### Required Argument
| Argument | Description | Example |
|----------|-------------|---------|
| `--url` | **Required**: Starting URL to begin scraping | `--url https://example.com` |

### Scraping Configuration
| Argument | Type | Default | Description | Example |
|----------|------|---------|-------------|---------|
| `--depth` | integer | 3 | Depth limit for crawling (how many levels deep to go) | `--depth 2` |
| `--threads` | integer | 5 | Number of concurrent threads for faster scraping | `--threads 8` |
| `--delay` | float | 0.5 | Delay between requests in seconds (prevents blocking) | `--delay 0.2` |

### Feature Toggles
| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--no-trafilatura` | flag | Disable advanced content extraction (uses fallback method) | `--no-trafilatura` |
| `--no-robots` | flag | Ignore robots.txt rules (use with caution) | `--no-robots` |
| `--embeddings` | flag | Generate text embeddings using sentence-transformers | `--embeddings` |

### Output Configuration
| Argument | Type | Default | Description | Example |
|----------|------|---------|-------------|---------|
| `--data-file` | string | scraped_data.json | Output file for scraped content | `--data-file my_data.json` |
| `--log-file` | string | scraped_log.txt | Output file for navigation log | `--log-file my_log.txt` |
| `--embeddings-file` | string | embeddings.json | Output file for embeddings (requires --embeddings) | `--embeddings-file my_embeddings.json` |

## Common Use Cases

### Quick Scrape
```cmd
python app1.py --url https://example.com
```

### Comprehensive Scrape with Embeddings
```cmd
python app1.py --url https://example.com --depth 4 --threads 10 --embeddings
```

### Fast Scrape (Lower Delay)
```cmd
python app1.py --url https://example.com --delay 0.1 --no-robots
```

### Custom Output Files
```cmd
python app1.py --url https://example.com --data-file website_data.json --log-file crawl_log.txt
```

### Help Command
```cmd
python app1.py --help
```

## Important Notes
- URLs must start with `http://` or `https://`
- Lower delays may trigger rate limiting or IP blocking
- Ignoring robots.txt (`--no-robots`) may violate website terms of service
- Embeddings require `sentence-transformers` library installed
- The scraper respects politeness delays by default to avoid overwhelming servers
