# Sitejabber Scraper Yapılandırma Dosyası
# Tüm sabitler, hedef şirketler ve gecikme ayarları burada tutulur.

# Sitejabber'ın inceleme sayfası için temel URL şablonu
BASE_URL = "https://www.sitejabber.com/reviews/{slug}?page={page}"

# Kazınacak hedef e-ticaret şirketlerinin slug listesi
TARGET_COMPANIES = [
    "amazon.com",
    "ebay.com",
    "etsy.com",
    "walmart.com",
    "aliexpress.com",
]

# Her şirket için kazınacak maksimum sayfa sayısı
MAX_PAGES_PER_COMPANY = 5

# Kazınan verilerin kaydedileceği CSV dosyasının yolu
OUTPUT_FILE = "data/reviews.csv"

# İstekler arasındaki minimum bekleme süresi (saniye)
MIN_DELAY = 3.0

# İstekler arasındaki maksimum bekleme süresi (saniye)
MAX_DELAY = 6.0

# Sayfanın JS içeriğini yüklemesi için beklenen süre (saniye)
PAGE_LOAD_WAIT = 8

# Yorum elementlerinin DOM'a gelmesi için maksimum bekleme (saniye)
ELEMENT_WAIT_TIMEOUT = 20

# Rotasyonda kullanılacak User-Agent listesi
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",

    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",

    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.4.1 Safari/605.1.15",

    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",

    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",
]
