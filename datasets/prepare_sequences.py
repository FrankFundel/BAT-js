import numpy as np

germanBats = {
    "Rhinolophus ferrumequinum": 0,
    "Rhinolophus hipposideros": 1,
    "Myotis daubentonii": 2,
    "Myotis brandtii": 3,
    "Myotis mystacinus": 4,
    "Myotis emarginatus": 5,
    "Myotis nattereri": 6,
    #"Myotis bechsteinii": 7,
    "Myotis myotis": 7,
    "Myotis dasycneme": 8,
    "Nyctalus noctula": 9,
    "Nyctalus leisleri": 10,
    "Pipistrellus pipistrellus": 11,
    "Pipistrellus nathusii": 12,
    "Pipistrellus kuhlii": 13,
    "Eptesicus serotinus": 14,
    "Eptesicus nilssonii": 15,
    #"Plecotus auritus": 16,
    #"Plecotus austriacus": 16,
    #"Barbastella barbastellus": 16,
    #"Tadarida teniotis": 16,
    "Miniopterus schreibersii": 16,
    #"Hypsugo savii": 18,
    "Vespertilio murinus": 17,
}

def slideWindow(a, size, step):
    b = []
    i = 0
    pos = 0
    while pos + size < len(a):
        pos = int(i * step)
        tile = a[pos : pos + size]
        b.append(tile)
        i+=1
    return b

def getSequences(spectrogram, patch_len, patch_skip, seq_len, seq_skip):
    tiles = slideWindow(spectrogram, patch_len, patch_skip)[:-1] # last one is not full
    if len(tiles) <= seq_len:
        for _ in range(seq_len - len(tiles)):
            tiles.append(np.zeros_like(tiles[0]))
        return [tiles]
    sequences = slideWindow(tiles, seq_len, seq_skip)[:-1] # last one is not full
    return sequences
