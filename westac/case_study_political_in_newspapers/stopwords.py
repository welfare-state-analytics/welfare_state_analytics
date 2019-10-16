import nltk

fredrik_stopwords = [
    '""', '``',
    'aderton',
    'adertonde',
    'adjö',
    'aldrig',
    'alla',
    'allas',
    'allt',
    'alltid',
    'alltså',
    'andra',
    'andras',
    'annan',
    'annat',
    'artonde',
    'artonn',
    'att',
    'av',
    'bakom',
    'bara',
    'behöva',
    'behövas',
    'behövde',
    'behövt',
    'beslut',
    'beslutat',
    'beslutit',
    'bland',
    'blev',
    'bli',
    'blir',
    'blivit',
    'bort',
    'borta',
    'bra',
    'bäst',
    'bättre',
    'båda',
    'bådas',
    'dag',
    'dagar',
    'dagarna',
    'dagen',
    'de',
    'del',
    'delen',
    'dem',
    'den',
    'denna',
    'deras',
    'dess',
    'dessa',
    'det',
    'detta',
    'dig',
    'din',
    'dina',
    'dit',
    'ditt',
    'dock',
    'du',
    'där',
    'därför',
    'då',
    'efter',
    'eftersom',
    'ej',
    'elfte',
    'eller',
    'elva',
    'en',
    'enkel',
    'enkelt',
    'enkla',
    'enligt',
    'er',
    'era',
    'ert',
    'ett',
    'ettusen',
    'fanns',
    'fem',
    'femte',
    'femtio',
    'femtionde',
    'femton',
    'femtonde',
    'fick',
    'fin',
    'finnas',
    'finns',
    'fjorton',
    'fjortonde',
    'fjärde',
    'fler',
    'flera',
    'flesta',
    'fram',
    'framför',
    'från',
    'fyra',
    'fyrtio',
    'fyrtionde',
    'få',
    'får',
    'fått',
    'följande',
    'för',
    'före',
    'förlåt',
    'förra',
    'första',
    'genast',
    'genom',
    'gick',
    'gjorde',
    'gjort',
    'god',
    'goda',
    'godare',
    'godast',
    'gott',
    'gälla',
    'gäller',
    'gällt',
    'gärna',
    'gå',
    'går',
    'gått',
    'gör',
    'göra',
    'ha',
    'hade',
    'haft',
    'han',
    'hans',
    'har',
    'heller',
    'hellre',
    'helst',
    'helt',
    'henne',
    'hennes',
    'hit',
    'hon',
    'honom',
    'hundra',
    'hundraen',
    'hundraett',
    'hur',
    'här',
    'hög',
    'höger',
    'högre',
    'högst',
    'i',
    'ibland',
    'icke',
    'idag',
    'igen',
    'igår',
    'imorgon',
    'in',
    'inför',
    'inga',
    'ingen',
    'ingenting',
    'inget',
    'innan',
    'inne',
    'inom',
    'inte',
    'inuti',
    'ja',
    'jag',
    'ju',
    'jämfört',
    'kan',
    'kanske',
    'knappast',
    'kom',
    'komma',
    'kommer',
    'kommit',
    'kr',
    'kunde',
    'kunna',
    'kunnat',
    'kvar',
    'legat',
    'ligga',
    'ligger',
    'lika',
    'likställd',
    'likställda',
    'lilla',
    'lite',
    'liten',
    'litet',
    'länge',
    'längre',
    'längst',
    'lätt',
    'lättare',
    'lättast',
    'långsam',
    'långsammare',
    'långsammast',
    'långsamt',
    'långt',
    'man',
    'med',
    'mellan',
    'men',
    'mer',
    'mera',
    'mest',
    'mig',
    'min',
    'mina',
    'mindre',
    'minst',
    'mitt',
    'mittemot',
    'mot',
    'mycket',
    'många',
    'måste',
    'möjlig',
    'möjligen',
    'möjligt',
    'möjligtvis',
    'ned',
    'nederst',
    'nedersta',
    'nedre',
    'nej',
    'ner',
    'ni',
    'nio',
    'nionde',
    'nittio',
    'nittionde',
    'nitton',
    'nittonde',
    'nog',
    'noll',
    'nr',
    'nu',
    'nummer',
    'när',
    'nästa',
    'någon',
    'någonting',
    'något',
    'några',
    'nödvändig',
    'nödvändiga',
    'nödvändigt',
    'nödvändigtvis',
    'och',
    'också',
    'ofta',
    'oftast',
    'olika',
    'olikt',
    'om',
    'oss',
    'på',
    'rakt',
    'redan',
    'rätt',
    'sade',
    'sagt',
    'samma',
    'sedan',
    'senare',
    'senast',
    'sent',
    'sex',
    'sextio',
    'sextionde',
    'sexton',
    'sextonde',
    'sig',
    'sin',
    'sina',
    'sist',
    'sista',
    'siste',
    'sitt',
    'sitta',
    'sju',
    'sjunde',
    'sjuttio',
    'sjuttionde',
    'sjutton',
    'sjuttonde',
    'själv',
    'sjätte',
    'ska',
    'skall',
    'skulle',
    'slutligen',
    'små',
    'smått',
    'snart',
    'som',
    'stor',
    'stora',
    'stort',
    'större',
    'störst',
    'säga',
    'säger',
    'sämre',
    'sämst',
    'så',
    'sådan',
    'sådana',
    'sådant',
    'tack',
    'tidig',
    'tidigare',
    'tidigast',
    'tidigt',
    'till',
    'tills',
    'tillsammans',
    'tio',
    'tionde',
    'tjugo',
    'tjugoen',
    'tjugoett',
    'tjugonde',
    'tjugotre',
    'tjugotvå',
    'tjungo',
    'tolfte',
    'tolv',
    'tre',
    'tredje',
    'trettio',
    'trettionde',
    'tretton',
    'trettonde',
    'två',
    'tvåhundra',
    'under',
    'upp',
    'ur',
    'ursäkt',
    'ut',
    'utan',
    'utanför',
    'ute',
    'vad',
    'var',
    'vara',
    'varför',
    'varifrån',
    'varit',
    'varje',
    'varken',
    'vars',
    'varsågod',
    'vart',
    'vem',
    'vems',
    'verkligen',
    'vi',
    'vid',
    'vidare',
    'viktig',
    'viktigare',
    'viktigast',
    'viktigt',
    'vilka',
    'vilkas',
    'vilken',
    'vilket',
    'vill',
    'vänster',
    'vänstra',
    'värre',
    'vår',
    'våra',
    'vårt',
    'än',
    'ännu',
    'är',
    'även',
    'åt',
    'åtminstone',
    'åtta',
    'åttio',
    'åttionde',
    'åttonde',
    'över',
    'övermorgon',
    'överst',
    'övre'
]

stopwords = set(nltk.corpus.stopwords.words('swedish'))\
    .union({ "politisk", "politiska", "politiskt" })\
    .union(set(fredrik_stopwords))

