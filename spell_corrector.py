import re

dictionary = {
    'Dto, d2': 'Dito',
    'npo, napo': 'na po',
    'ty': 'thank you',
    'p': 'pa',
    'Jr, Jr\'s, jr': 'Junior',
    'pngwa': 'pinagawa',
    'niyu, nyo, nyu': 'niyo',
    'kyo, kayu, kau': 'kayo',
    'c': 'si',
    'pk, pki': 'paki',
    'txt, TX': 'text',
    'ung, yng': 'yung',
    'ska, Ska': 'tsaka',
    'ky': 'kay',
    'mg': 'Mag',
    'knina': 'kanina',
    'kpa': 'ka pa',
    'tnwagan': 'tinawagan',
    'm': 'mo',
    'n': 'ni',
    'San': 'Saan',
    'bka': 'baka',
    'nmn, nman': 'naman',
    'allas': 'alas',
    'nkkahiya': 'nakakahiya',
    'kpot': 'kapit',
    'bhay': 'bahay',
    'Wla': 'Wala',
    'youuuuuuu': 'you',
    'loveee': 'love',
    'san': 'saan',
    'naaa, n': 'na',
    'mba': 'mo ba',
    'nak': 'anak',
    'Bkit': 'Bakit',
    'dmo': 'di mo',
    'kna, kana': 'ka na',
    'uwe': 'uwi',
    'D': 'Edi',
    'Cno': 'Sino',
    'Nkapila': 'Nakapila',
    'kb': 'ka ba',
    'b': 'ba',
    'jan': 'dyan',
    'db': 'hindi ba',
    'Di': 'Hindi',
    'mabalik': 'maibalik',
    'lowbat, low bat': 'low battery',
    'Atehh': 'Ate',
    'nlng': 'na lang',
    'makauwe': 'makauwi',
    'kelangan, kilangan': 'kailangan',
    'pira': 'pera',
    'dw': 'daw',
    'U, u': 'YOU',
    'Ge': 'Sige',
    'Otw, ontheway': 'On the way',
    'Hellow': 'Hello',
    'Gooday': 'Good day',
    'kakaalis': 'kaaalis',
    'Po': 'po',
    'don': 'doon',
    'Gud': 'Good',
    'pagdting': 'pagdating',
    'syaa': 'siya',
    'pnta': 'punta',
    'pls': 'please',
    'magkakaron': 'magkakaroon',
    'pde, pwd': 'pwede',
    'till': 'until',
    'panu': 'paano',
    'pic': 'picture',
    'facrshield': 'faceshield',
    'ur': 'your',
    'Ty, ty': 'Thank you',
    'anu': 'ano',
    'linagay': 'inilagay',
    'gsing': 'gising',
    'Yng': 'Yung',
    'prblema': 'problema',
    'Tnx': 'Thanks',
    'Pki': 'Paki',
    'nya': 'niya',
    '4ward': 'forward',
    'bng': 'bang',
    'd': 'hindi',
    'c': 'si',
    'w': 'with',
    "you're": 'you are',
    "you'll": 'you will',
    'acct': 'account',
    'app': 'application',
    'ref': 'reference'
}



number_dictionary = {
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'zero': '0'
}


# Combine the two dictionaries
combined_dictionary = {**dictionary, **number_dictionary}

'''
def custom_vectorizer(text):
    for key, value in combined_vector.items():
        # Split the key into variations
        variations = [v.strip() for v in key.split(',')]
        
        # Create a regular expression pattern with word boundaries for each variation
        pattern = '|'.join(r'\b' + re.escape(v) + r'\b' for v in variations)
        
        # Use re.sub with the pattern for replacement
        text = re.sub(pattern, value, text, flags=re.IGNORECASE)
   
    return text
'''






