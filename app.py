import sys
import json
import sqlite3
import threading
import hashlib
import time
from datetime import datetime
from urllib.parse import urlparse, urljoin
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtGui import *
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import os

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class SearchIndex:
    
    def __init__(self, db_path='search_engine.db'):
        self.db_path = db_path
        self.init_database()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=stopwords.words('english') + stopwords.words('russian'),
            ngram_range=(1, 2)
        )
        self.doc_vectors = None
        self.doc_urls = []
        self.doc_contents = []
        self.build_vectors_from_database()  
        
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Таблица для веб-страниц
        c.execute('''
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                content TEXT,
                clean_content TEXT,
                description TEXT,
                indexed_at TIMESTAMP,
                page_rank REAL DEFAULT 1.0,
                link_count INTEGER DEFAULT 0,
                indexed_count INTEGER DEFAULT 0
            )
        ''')
        
        # Инвертированный индекс
        c.execute('''
            CREATE TABLE IF NOT EXISTS inverted_index (
                word TEXT,
                page_id INTEGER,
                frequency INTEGER,
                positions TEXT,
                FOREIGN KEY (page_id) REFERENCES pages (id)
            )
        ''')
        
        # Ссылки между страницами (для PageRank)
        c.execute('''
            CREATE TABLE IF NOT EXISTS links (
                from_page_id INTEGER,
                to_page_id INTEGER,
                FOREIGN KEY (from_page_id) REFERENCES pages (id),
                FOREIGN KEY (to_page_id) REFERENCES pages (id)
            )
        ''')
        
        # История поисковых запросов
        c.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                timestamp TIMESTAMP,
                results_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def build_vectors_from_database(self):

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            c.execute('SELECT id, clean_content FROM pages WHERE clean_content != ""')
            results = c.fetchall()
            
            if results:
                self.doc_urls = [row[0] for row in results]  # page_ids
                self.doc_contents = [row[1] for row in results]
                
                if len(self.doc_contents) > 0:
                    self.doc_vectors = self.vectorizer.fit_transform(self.doc_contents)
                else:
                    self.doc_vectors = None
            else:
                self.doc_vectors = None
                self.doc_urls = []
                self.doc_contents = []
                
        except Exception as e:
            print(f"Ошибка при загрузке индекса: {e}")
            self.doc_vectors = None
        finally:
            conn.close()
    
    def clean_text(self, text):
        """Очистка текста от HTML и нормализация"""
        if not text:
            return ""
        
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Токенизация текста"""
        try:
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english') + stopwords.words('russian'))
            tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
            return tokens
        except:
            return text.split()  
    
    def add_page(self, url, title, content, description=""):
        """Добавление страницы в индекс"""
        clean_content = self.clean_text(content)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            c.execute('''
                INSERT OR REPLACE INTO pages 
                (url, title, content, clean_content, description, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (url, title, content, clean_content, description, datetime.now()))

            c.execute('SELECT id FROM pages WHERE url = ?', (url,))
            result = c.fetchone()
            page_id = result[0] if result else c.lastrowid

            c.execute('DELETE FROM inverted_index WHERE page_id = ?', (page_id,))

            tokens = self.tokenize(clean_content)
            

            word_positions = {}
            for i, token in enumerate(tokens):
                if token not in word_positions:
                    word_positions[token] = []
                word_positions[token].append(i)
            

            for word, positions in word_positions.items():
                c.execute('''
                    INSERT INTO inverted_index (word, page_id, frequency, positions)
                    VALUES (?, ?, ?, ?)
                ''', (word, page_id, len(positions), json.dumps(positions)))
            
            conn.commit()

            self.build_vectors_from_database()
            
            return page_id
            
        except Exception as e:
            print(f"Ошибка при индексации {url}: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def extract_links(self, html, base_url):
        """Извлечение ссылок из HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href:

                    absolute_url = urljoin(base_url, href)

                    if absolute_url.startswith(('http://', 'https://')):

                        parsed = urlparse(absolute_url)
                        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                        if clean_url not in links:
                            links.append(clean_url)
            
            return links
        except Exception as e:
            print(f"Ошибка при извлечении ссылок: {e}")
            return []
    
    def search(self, query, page=1, results_per_page=10):
        """Поиск по индексу с ранжированием"""
        start_time = time.time()
        clean_query = self.clean_text(query)

        self.save_search_history(query)

        if self.doc_vectors is None or len(self.doc_contents) == 0 or not clean_query:
            return {
                'query': query,
                'results': [],
                'total_results': 0,
                'page': page,
                'results_per_page': results_per_page,
                'time': 0
            }
        
        try:

            query_vector = self.vectorizer.transform([clean_query])
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            

            top_indices = similarities.argsort()[::-1]
            
            results = []
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            for idx in top_indices[:results_per_page * 3]:  # Берем больше для последующей фильтрации
                if similarities[idx] > 0.01:  # Минимальный порог схожести
                    page_id = self.doc_urls[idx]
                    c.execute('''
                        SELECT url, title, description, page_rank 
                        FROM pages WHERE id = ?
                    ''', (page_id,))
                    row = c.fetchone()
                    
                    if row:
                        url, title, desc, page_rank = row
                        
                        # Комбинированный рейтинг (TF-IDF * PageRank)
                        combined_score = similarities[idx] * (page_rank or 1.0)
                        
                        # Получаем сниппет
                        c.execute('SELECT content FROM pages WHERE id = ?', (page_id,))
                        content_row = c.fetchone()
                        snippet = self.generate_snippet(content_row[0] if content_row else "", query)
                        
                        results.append({
                            'url': url,
                            'title': title or url,
                            'snippet': snippet,
                            'description': desc or snippet[:150],
                            'score': float(combined_score),
                            'similarity': float(similarities[idx])
                        })
            
            conn.close()
            
            # Сортируем по комбинированному рейтингу
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Пагинация
            total_results = len(results)
            start_idx = (page - 1) * results_per_page
            end_idx = start_idx + results_per_page
            
            search_time = time.time() - start_time
            
            return {
                'query': query,
                'results': results[start_idx:end_idx],
                'total_results': total_results,
                'page': page,
                'results_per_page': results_per_page,
                'time': search_time
            }
            
        except Exception as e:
            print(f"Ошибка при поиске: {e}")
            return {
                'query': query,
                'results': [],
                'total_results': 0,
                'page': page,
                'results_per_page': results_per_page,
                'time': time.time() - start_time
            }
    
    def generate_snippet(self, content, query, max_length=200):
        """Генерация сниппета с подсветкой запроса"""
        if not content:
            return "Нет описания"

        content_lower = content.lower()
        query_words = [q.lower() for q in query.split() if len(q) > 2]
        
        if not query_words:

            snippet = content[:max_length]
            if len(content) > max_length:
                snippet += "..."
            return snippet
        
        best_pos = -1
        best_score = 0
        
        
        for i in range(0, len(content_lower), 100):
            score = 0
            for word in query_words:
                if word in content_lower[i:i+300]:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_pos = i
        
        if best_pos == -1:

            snippet = content[:max_length]
        else:

            start = max(0, best_pos - 50)
            end = min(len(content), start + max_length)
            snippet = content[start:end]

            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
        
        return snippet
    
    def save_search_history(self, query):
        """Сохранение истории поиска"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:

            results_count = 0
            if self.doc_vectors is not None:
                clean_query = self.clean_text(query)
                if clean_query:
                    query_vector = self.vectorizer.transform([clean_query])
                    similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
                    results_count = np.sum(similarities > 0.01)
            
            c.execute('''
                INSERT INTO search_history (query, timestamp, results_count)
                VALUES (?, ?, ?)
            ''', (query, datetime.now(), int(results_count)))
            
            conn.commit()
        except Exception as e:
            print(f"Ошибка при сохранении истории: {e}")
        finally:
            conn.close()
    
    def calculate_page_rank(self, iterations=10, damping=0.85):
        """Вычисление PageRank для страниц"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            c.execute('SELECT id FROM pages')
            pages = [row[0] for row in c.fetchall()]
            
            if not pages:
                return
            
            page_rank = {page: 1.0 for page in pages}
            
            c.execute('SELECT from_page_id, to_page_id FROM links')
            links = c.fetchall()
            
            graph = {page: [] for page in pages}
            link_counts = {page: 0 for page in pages}
            
            for from_id, to_id in links:
                if from_id in graph and to_id in graph:
                    graph[from_id].append(to_id)
                    link_counts[from_id] = link_counts.get(from_id, 0) + 1
            
            for _ in range(iterations):
                new_rank = {}
                
                for page in pages:
                    rank_sum = 0
                    
                    for from_page in pages:
                        if page in graph.get(from_page, []):
                            if link_counts.get(from_page, 0) > 0:
                                rank_sum += page_rank[from_page] / link_counts[from_page]
                    
                    new_rank[page] = (1 - damping) + damping * rank_sum
                
                page_rank = new_rank
            
            for page_id, rank in page_rank.items():
                c.execute('UPDATE pages SET page_rank = ? WHERE id = ?', (rank, page_id))
            
            conn.commit()
            
        except Exception as e:
            print(f"Ошибка при расчете PageRank: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_statistics(self):
        """Получение статистики поисковой системы"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            c.execute('SELECT COUNT(*) FROM pages')
            total_pages = c.fetchone()[0] or 0
            
            c.execute('SELECT COUNT(DISTINCT word) FROM inverted_index')
            unique_words_result = c.fetchone()
            unique_words = unique_words_result[0] if unique_words_result else 0
            
            c.execute('SELECT COUNT(*) FROM search_history')
            total_searches_result = c.fetchone()
            total_searches = total_searches_result[0] if total_searches_result else 0
            
            c.execute('SELECT COUNT(*) FROM links')
            total_links_result = c.fetchone()
            total_links = total_links_result[0] if total_links_result else 0
            
            return {
                'total_pages': total_pages,
                'unique_words': unique_words,
                'total_searches': total_searches,
                'total_links': total_links,
                'index_size_mb': self.get_database_size()
            }
        except Exception as e:
            print(f"Ошибка при получении статистики: {e}")
            return {
                'total_pages': 0,
                'unique_words': 0,
                'total_searches': 0,
                'total_links': 0,
                'index_size_mb': 0
            }
        finally:
            conn.close()
    
    def get_database_size(self):
        """Получение размера базы данных в МБ"""
        try:
            if os.path.exists(self.db_path):
                return os.path.getsize(self.db_path) / (1024 * 1024)
        except:
            pass
        return 0

class WebCrawler(QThread):
    """Веб-краулер для индексации страниц"""
    
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal()
    
    def __init__(self, search_index):
        super().__init__()
        self.search_index = search_index
        self.urls_to_crawl = []
        self.crawled_urls = set()
        self.max_pages = 100
        self.running = False
        self.lock = threading.Lock()
        
    def start_crawling(self, start_urls, max_pages=100):
        """Начало краулинга"""
        self.urls_to_crawl = list(start_urls)
        self.max_pages = max_pages
        self.crawled_urls.clear()
        self.running = True
        self.start()
    
    def run(self):
        """Основной цикл краулинга"""
        pages_crawled = 0
        
        while self.running and self.urls_to_crawl and pages_crawled < self.max_pages:
            url = self.urls_to_crawl.pop(0)
            
            with self.lock:
                if url in self.crawled_urls:
                    continue
            
            try:
                self.progress.emit(pages_crawled, self.max_pages, url)
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                title_tag = soup.find('title')
                title = title_tag.string if title_tag else url
                
                description = ""
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    description = meta_desc['content']
                
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)
                
                self.search_index.add_page(url, str(title)[:500], content[:10000], description[:500])
                
                if pages_crawled < self.max_pages * 0.8:  
                    new_links = self.search_index.extract_links(str(soup), url)
                    
                    with self.lock:
                        for link in new_links[:5]:  
                            if link not in self.crawled_urls and link not in self.urls_to_crawl:
                                self.urls_to_crawl.append(link)
                
                with self.lock:
                    self.crawled_urls.add(url)
                
                pages_crawled += 1
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Ошибка при краулинге {url}: {e}")
                continue
        
        self.running = False
        self.finished.emit()
    
    def stop(self):
        """Остановка краулинга"""
        self.running = False

class SearchResultsWidget(QWidget):
    """Виджет для отображения результатов поиска"""
    
    def __init__(self, browser):
        super().__init__()
        self.browser = browser
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(self.info_label)
        
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.on_result_clicked)
        self.results_list.setStyleSheet("""
            QListWidget {
                border: none;
                background: white;
            }
            QListWidget::item {
                border-bottom: 1px solid #eee;
                padding: 10px;
            }
            QListWidget::item:hover {
                background: #f5f5f5;
            }
        """)
        layout.addWidget(self.results_list)
        
        
        pagination_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("← Назад")
        self.prev_btn.clicked.connect(self.prev_page)
        self.prev_btn.setEnabled(False)
        
        self.next_btn = QPushButton("Вперед →")
        self.next_btn.clicked.connect(self.next_page)
        self.next_btn.setEnabled(False)
        
        self.page_label = QLabel("Страница 1")
        
        pagination_layout.addWidget(self.prev_btn)
        pagination_layout.addWidget(self.page_label)
        pagination_layout.addWidget(self.next_btn)
        pagination_layout.addStretch()
        
        layout.addLayout(pagination_layout)
        self.setLayout(layout)
        
        self.current_page = 1
        self.current_query = ""
        self.total_results = 0
        
    def show_results(self, search_results):
        """Отображение результатов поиска"""
        self.current_query = search_results['query']
        self.current_page = search_results['page']
        self.total_results = search_results['total_results']
        
        time_text = f"{search_results['time']:.2f}" if search_results['time'] > 0 else "0.00"
        self.info_label.setText(
            f"Найдено результатов: {self.total_results} • Время: {time_text} сек."
        )
        
        self.results_list.clear()
        
        for result in search_results['results']:
            item = QListWidgetItem()
            
            widget = QWidget()
            layout = QVBoxLayout()
            
            title_text = result["title"] if result["title"] else result["url"]
            title_label = QLabel(f'<a href="{result["url"]}" style="color: #1a0dab; text-decoration: none; font-size: 18px;">{title_text}</a>')
            title_label.setOpenExternalLinks(False)
            title_label.linkActivated.connect(self.browser.load_url)
            layout.addWidget(title_label)
            
            url_label = QLabel(f'<span style="color: #006621; font-size: 14px;">{result["url"]}</span>')
            layout.addWidget(url_label)
            
            snippet_text = result["snippet"] if result["snippet"] else "Нет описания"
            snippet_label = QLabel(f'<span style="color: #545454; font-size: 13px;">{snippet_text}</span>')
            snippet_label.setWordWrap(True)
            layout.addWidget(snippet_label)
            
            if "score" in result:
                score_label = QLabel(f'<span style="color: #999; font-size: 11px;">Рейтинг: {result["score"]:.4f}</span>')
                layout.addWidget(score_label)
            
            widget.setLayout(layout)
            item.setSizeHint(widget.sizeHint())
            
            self.results_list.addItem(item)
            self.results_list.setItemWidget(item, widget)
        
        self.update_pagination()
        
    def update_pagination(self):
        """Обновление кнопок пагинации"""
        results_per_page = 10
        total_pages = max(1, (self.total_results + results_per_page - 1) // results_per_page)
        
        self.prev_btn.setEnabled(self.current_page > 1)
        self.next_btn.setEnabled(self.current_page < total_pages)
        self.page_label.setText(f"Страница {self.current_page} из {total_pages}")
        
    def prev_page(self):
        """Переход на предыдущую страницу"""
        if self.current_page > 1:
            self.current_page -= 1
            self.browser.perform_search(self.current_query, self.current_page)
            
    def next_page(self):
        """Переход на следующую страницу"""
        self.current_page += 1
        self.browser.perform_search(self.current_query, self.current_page)
        
    def on_result_clicked(self, item):
        """Обработка клика по результату"""
        index = self.results_list.row(item)

class BrowserWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.search_index = SearchIndex()
        self.crawler = WebCrawler(self.search_index)
        self.crawler.progress.connect(self.update_crawler_progress)
        self.crawler.finished.connect(self.crawler_finished)
        
        self.current_mode = "browser"
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Независимая поисковая система v1.0')
        self.setGeometry(100, 100, 1400, 900)
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        control_group = QGroupBox("Управление")
        control_layout = QVBoxLayout()
        
        self.browser_mode_btn = QPushButton("Браузер")
        self.browser_mode_btn.clicked.connect(lambda: self.switch_mode("browser"))
        self.browser_mode_btn.setCheckable(True)
        self.browser_mode_btn.setChecked(True)
        
        self.search_mode_btn = QPushButton("Поисковая система")
        self.search_mode_btn.clicked.connect(lambda: self.switch_mode("search"))
        self.search_mode_btn.setCheckable(True)
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.browser_mode_btn)
        mode_layout.addWidget(self.search_mode_btn)
        control_layout.addLayout(mode_layout)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Введите поисковый запрос...")
        self.search_input.returnPressed.connect(self.on_search)
        control_layout.addWidget(self.search_input)
        
        self.search_btn = QPushButton("Искать")
        self.search_btn.clicked.connect(self.on_search)
        control_layout.addWidget(self.search_btn)
        
        crawler_group = QGroupBox("Индексатор сайтов")
        crawler_layout = QVBoxLayout()
        
        self.crawl_input = QLineEdit()
        self.crawl_input.setPlaceholderText("Введите URL для индексации (через запятую)")
        crawler_layout.addWidget(self.crawl_input)
        
        self.crawl_btn = QPushButton("Начать индексацию")
        self.crawl_btn.clicked.connect(self.start_crawling)
        crawler_layout.addWidget(self.crawl_btn)
        
        self.stop_crawl_btn = QPushButton("Остановить")
        self.stop_crawl_btn.clicked.connect(self.stop_crawling)
        self.stop_crawl_btn.setEnabled(False)
        crawler_layout.addWidget(self.stop_crawl_btn)
        
        self.crawl_progress = QProgressBar()
        crawler_layout.addWidget(self.crawl_progress)
        
        self.crawl_status = QLabel("Готов к работе")
        crawler_layout.addWidget(self.crawl_status)
        
        crawler_group.setLayout(crawler_layout)
        control_layout.addWidget(crawler_group)
        
        stats_group = QGroupBox("Статистика")
        stats_layout = QVBoxLayout()
        
        self.stats_label = QLabel()
        self.update_stats()
        stats_layout.addWidget(self.stats_label)
        
        refresh_stats_btn = QPushButton("Обновить статистику")
        refresh_stats_btn.clicked.connect(self.update_stats)
        stats_layout.addWidget(refresh_stats_btn)
        
        stats_group.setLayout(stats_layout)
        control_layout.addWidget(stats_group)
        
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        left_layout.addStretch()
        
        self.right_widget = QStackedWidget()
        
        self.browser_widget = QWidget()
        browser_layout = QVBoxLayout(self.browser_widget)
        
        nav_layout = QHBoxLayout()
        
        self.back_btn = QPushButton("←")
        self.back_btn.clicked.connect(self.go_back)
        
        self.forward_btn = QPushButton("→")
        self.forward_btn.clicked.connect(self.go_forward)
        
        self.reload_btn = QPushButton("↻")
        self.reload_btn.clicked.connect(self.reload_page)
        
        self.url_bar = QLineEdit()
        self.url_bar.returnPressed.connect(self.navigate_to_url)
        self.url_bar.setText("about:blank")
        
        nav_layout.addWidget(self.back_btn)
        nav_layout.addWidget(self.forward_btn)
        nav_layout.addWidget(self.reload_btn)
        nav_layout.addWidget(self.url_bar)
        
        browser_layout.addLayout(nav_layout)
        
        self.web_browser = QWebEngineView()
        self.web_browser.urlChanged.connect(self.update_url_bar)
        self.web_browser.loadFinished.connect(self.on_page_loaded)
        self.web_browser.setUrl(QUrl("about:blank"))
        
        browser_layout.addWidget(self.web_browser)
        
        self.search_results_widget = SearchResultsWidget(self)
        
        self.right_widget.addWidget(self.browser_widget)
        self.right_widget.addWidget(self.search_results_widget)
        
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(self.right_widget)
        self.splitter.setSizes([350, 1050])
        
        self.create_menu()
        
        QTimer.singleShot(1000, self.show_welcome_message)
        
    def show_welcome_message(self):
        """Показать приветственное сообщение"""
        welcome_text = """
        <h2>Добро пожаловать в независимую поисковую систему!</h2>
        <p><b>Для начала работы:</b></p>
        <ol>
        <li>Введите URL сайтов для индексации (например: example.com)</li>
        <li>Нажмите "Начать индексацию"</li>
        <li>После индексации используйте поле поиска для поиска</li>
        <li>Переключайтесь между режимами "Браузер" и "Поисковая система"</li>
        </ol>
        <p>Система создает полностью независимый поисковый индекс на вашем компьютере.</p>
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Добро пожаловать!")
        msg.setText(welcome_text)
        msg.setTextFormat(Qt.RichText)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()
        
    def create_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("Файл")
        
        index_page_action = QAction("Проиндексировать текущую страницу", self)
        index_page_action.triggered.connect(self.index_current_page)
        file_menu.addAction(index_page_action)
        
        export_index_action = QAction("Экспорт индекса", self)
        export_index_action.triggered.connect(self.export_index)
        file_menu.addAction(export_index_action)
        
        import_index_action = QAction("Импорт индекса", self)
        import_index_action.triggered.connect(self.import_index)
        file_menu.addAction(import_index_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        search_menu = menubar.addMenu("Поиск")
        
        clear_index_action = QAction("Очистить индекс", self)
        clear_index_action.triggered.connect(self.clear_index)
        search_menu.addAction(clear_index_action)
        
        recalc_pagerank_action = QAction("Пересчитать PageRank", self)
        recalc_pagerank_action.triggered.connect(self.recalculate_pagerank)
        search_menu.addAction(recalc_pagerank_action)
        
        view_menu = menubar.addMenu("Вид")
        
        show_stats_action = QAction("Показать статистику", self)
        show_stats_action.triggered.connect(self.show_statistics_window)
        view_menu.addAction(show_stats_action)
        
    def switch_mode(self, mode):
        """Переключение между режимами браузера и поиска"""
        self.current_mode = mode
        
        if mode == "browser":
            self.browser_mode_btn.setChecked(True)
            self.search_mode_btn.setChecked(False)
            self.right_widget.setCurrentWidget(self.browser_widget)
        else:
            self.browser_mode_btn.setChecked(False)
            self.search_mode_btn.setChecked(True)
            self.right_widget.setCurrentWidget(self.search_results_widget)
            
    def on_search(self):
        """Обработка поискового запроса"""
        query = self.search_input.text().strip()
        if query:
            self.switch_mode("search")
            self.perform_search(query)
            
    def perform_search(self, query, page=1):
        """Выполнение поиска"""
        search_results = self.search_index.search(query, page)
        self.search_results_widget.show_results(search_results)
        
    def start_crawling(self):
        """Запуск индексации сайтов"""
        urls_text = self.crawl_input.text().strip()
        if not urls_text:
            QMessageBox.warning(self, "Ошибка", "Введите хотя бы один URL")
            return
            
        urls = [url.strip() for url in urls_text.split(',')]
        valid_urls = []
        
        for url in urls:
            if url.startswith(('http://', 'https://')):
                valid_urls.append(url)
            else:
                valid_urls.append('http://' + url)
                
        if not valid_urls:
            QMessageBox.warning(self, "Ошибка", "Нет валидных URL")
            return
            
        self.crawl_btn.setEnabled(False)
        self.stop_crawl_btn.setEnabled(True)
        self.crawl_status.setText("Запуск индексации...")
        self.crawl_progress.setMaximum(100)
        self.crawl_progress.setValue(0)
        
        self.crawler.start_crawling(valid_urls, max_pages=1000)
        
    def stop_crawling(self):
        """Остановка индексации"""
        self.crawler.stop()
        self.crawl_status.setText("Индексация остановлена")
        self.crawl_btn.setEnabled(True)
        self.stop_crawl_btn.setEnabled(False)
        
    def update_crawler_progress(self, current, total, url):
        """Обновление прогресса индексации"""
        self.crawl_progress.setMaximum(total)
        self.crawl_progress.setValue(current)
        short_url = url[:40] + "..." if len(url) > 40 else url
        self.crawl_status.setText(f"Индексация: {short_url}")
        
    def crawler_finished(self):
        """Завершение индексации"""
        self.crawl_progress.setValue(0)
        self.crawl_status.setText("Индексация завершена")
        self.crawl_btn.setEnabled(True)
        self.stop_crawl_btn.setEnabled(False)
        self.update_stats()
        
        self.search_index.calculate_page_rank()
        
        QMessageBox.information(self, "Готово", f"Индексация завершена! Проиндексировано {len(self.crawler.crawled_urls)} страниц.")
        
    def update_stats(self):
        """Обновление статистики"""
        stats = self.search_index.get_statistics()
        stats_text = f"""
        <b>Статистика поисковой системы:</b><br>
        • Проиндексировано страниц: {stats['total_pages']}<br>
        • Уникальных слов: {stats['unique_words']}<br>
        • Всего поисков: {stats['total_searches']}<br>
        • Ссылок в графе: {stats['total_links']}<br>
        • Размер индекса: {stats['index_size_mb']:.2f} МБ<br>
        """
        self.stats_label.setText(stats_text)
        
    def index_current_page(self):
        """Индексация текущей страницы в браузере"""
        current_url = self.web_browser.url().toString()
        if current_url == "about:blank" or not current_url.startswith("http"):
            QMessageBox.warning(self, "Ошибка", "Нет открытой веб-страницы для индексации")
            return
        
        def callback(content):
            url = self.web_browser.url().toString()
            title = self.web_browser.page().title()
            self.search_index.add_page(url, title, content)
            self.update_stats()
            QMessageBox.information(self, "Успех", f"Страница {url} проиндексирована!")
            
        self.web_browser.page().toPlainText(callback)
        
    def clear_index(self):
        """Очистка поискового индекса"""
        reply = QMessageBox.question(self, "Подтверждение", 
                                   "Вы уверены, что хотите очистить весь индекс?\nЭто действие нельзя отменить.",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                if os.path.exists('search_engine.db'):
                    os.remove('search_engine.db')
                
                self.search_index = SearchIndex()
                self.update_stats()
                
                QMessageBox.information(self, "Готово", "Индекс успешно очищен!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось очистить индекс: {str(e)}")
            
    def recalculate_pagerank(self):
        """Пересчет PageRank"""
        QMessageBox.information(self, "Информация", "Начался пересчет PageRank...")
        self.search_index.calculate_page_rank()
        QMessageBox.information(self, "Готово", "PageRank пересчитан!")
        
    def show_statistics_window(self):
        """Показать окно с подробной статистикой"""
        stats = self.search_index.get_statistics()
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Подробная статистика")
        dialog.setGeometry(300, 300, 500, 350)
        
        layout = QVBoxLayout()
        
        stats_text = f"""
        <h3>📊 Статистика поисковой системы</h3>
        <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
        <tr style="background-color: #f2f2f2;"><td><b>Параметр</b></td><td><b>Значение</b></td></tr>
        <tr><td>📄 Проиндексировано страниц</td><td align="right">{stats['total_pages']}</td></tr>
        <tr><td>🔤 Уникальных слов</td><td align="right">{stats['unique_words']}</td></tr>
        <tr><td>🔍 Всего поисковых запросов</td><td align="right">{stats['total_searches']}</td></tr>
        <tr><td>🔗 Ссылок в графе</td><td align="right">{stats['total_links']}</td></tr>
        <tr><td>💾 Размер базы данных</td><td align="right">{stats['index_size_mb']:.2f} МБ</td></tr>
        </table>
        """
        
        stats_label = QLabel(stats_text)
        stats_label.setTextFormat(Qt.RichText)
        layout.addWidget(stats_label)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        dialog.exec_()
        
    def export_index(self):
        """Экспорт индекса в файл"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Экспорт индекса", "", "JSON files (*.json);;All files (*.*)"
        )
        
        if file_name:
            try:
                conn = sqlite3.connect('search_engine.db')
                c = conn.cursor()
                
                c.execute('SELECT * FROM pages')
                pages = c.fetchall()
                
                c.execute('SELECT * FROM inverted_index')
                inverted_index = c.fetchall()
                
                data = {
                    'pages': pages,
                    'inverted_index': inverted_index,
                    'export_date': datetime.now().isoformat()
                }
                
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                QMessageBox.information(self, "Успех", f"Индекс экспортирован в {file_name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось экспортировать индекс: {str(e)}")
            finally:
                conn.close()
        
    def import_index(self):
        """Импорт индекса из файла"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Импорт индекса", "", "JSON files (*.json);;All files (*.*)"
        )
        
        if file_name:
            reply = QMessageBox.question(
                self, "Подтверждение",
                "Импорт индекса перезапишет текущий индекс. Продолжить?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    with open(file_name, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    QMessageBox.information(self, "Успех", "Импорт индекса выполнен!")
                    self.update_stats()
                    
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Не удалось импортировать индекс: {str(e)}")
        
    def navigate_to_url(self):
        url = self.url_bar.text().strip()
        if url:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            self.web_browser.setUrl(QUrl(url))
        
    def update_url_bar(self, q):
        self.url_bar.setText(q.toString())
        
    def go_back(self):
        self.web_browser.back()
        
    def go_forward(self):
        self.web_browser.forward()
        
    def reload_page(self):
        self.web_browser.reload()
        
    def on_page_loaded(self, success):
        if success:
            self.statusBar().showMessage("Страница загружена", 3000)
        else:
            self.statusBar().showMessage("Ошибка загрузки страницы", 3000)
        
    def load_url(self, url):
        """Загрузка URL из результатов поиска"""
        self.switch_mode("browser")
        self.web_browser.setUrl(QUrl(url))
        self.url_bar.setText(url)

class SearchEngineApp(QApplication):
    """Главное приложение поисковой системы"""
    
    def __init__(self, argv):
        super().__init__(argv)
        self.setApplicationName("Independent Search Engine")
        self.setApplicationVersion("1.0")
        
        # Устанавливаем стиль
        self.setStyle("Fusion")
        
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Highlight, QColor(76, 175, 80))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:checked {
                background-color: #2196F3;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 12px;
                background-color: white;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
            QLabel {
                font-size: 12px;
            }
            QListWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)

def main():
    app = SearchEngineApp(sys.argv)
    
    browser = BrowserWindow()
    browser.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()