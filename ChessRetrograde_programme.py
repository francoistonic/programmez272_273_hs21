#-*- coding: utf-8 -*-

# =============================================================================
# Sous-classe de chess.Board spécifique pour la finale Fou-Cavalier
# =============================================================================
import chess # échiquier modélisé avec les règles des échecs

# Sous-classe pour étudier la finale Fou+Cavalier (King-Bishop-Knight vs King)
class Board_KBNvK(chess.Board):

  # Cases pour le roi noir (symétries)
  bK_SQUARES = [chess.A1, chess.B1, chess.C1, chess.D1,
                chess.A2, chess.B2, chess.C2, chess.D2,
                chess.A3, chess.B3, chess.C3, chess.D3,
                chess.A4, chess.B4, chess.C4, chess.D4]

  # Création d'un échiquier avec une position Roi contre Roi+Fou+Cavalier
  # bK:black King, wK:white King, wB:white Bishop, wN:white Knight
  def __init__(self, bK=None, wK=None, wB=None, wN=None):
    # création d'un échiquier vide
    super().__init__(None)
    # ajout des pièces de la finale Fou+Cavalier
    if bK != None: self.set_piece_at(bK, chess.Piece(chess.KING, chess.BLACK))
    if wK != None: self.set_piece_at(wK, chess.Piece(chess.KING, chess.WHITE))
    if wB != None: self.set_piece_at(wB, chess.Piece(chess.BISHOP, chess.WHITE))
    if wN != None: self.set_piece_at(wN, chess.Piece(chess.KNIGHT, chess.WHITE))

  # Affichage de l'échiquier
  def display(self):
    print(self)

  # Transformation en position canonique par symétries horizontale et verticale
  def canonical(self):
    if chess.square_file(self.king(chess.BLACK)) > 3:
      self.apply_transform(chess.flip_horizontal)
    if chess.square_rank(self.king(chess.BLACK)) > 3:
      self.apply_transform(chess.flip_vertical)

  # Création d'une nouvelle position en jouant un coup
  def play_move(self, coup:chess.Move):
    board = self.copy()
    board.push(coup)
    board.canonical()
    return board

  # Calcul de l'index de la position (de 0 à 3812255)
  def index(self):
    # position des pièces
    bK = self.king(chess.BLACK)
    wK = self.king(chess.WHITE)
    wB = list(self.pieces(chess.BISHOP, chess.WHITE))[0]
    wN = list(self.pieces(chess.KNIGHT, chess.WHITE))[0]
    # ajustement pour éviter les positions avec 2 pièces sur la même case
    wN = wN - (wN > wB) - (wN > wK) - (wN > bK)
    wB = wB - (wB > wK) - (wB > bK)
    wK = wK - (wK > bK)
    # calcul de l'index
    return 63*62*61*self.bK_SQUARES.index(bK) + 62*61*wK + 61*wB + wN

  # Itérateur sur les 16*63*62*61=3812256 positions canoniques
  @classmethod
  def canonical_boards(cls):
      # Boucle sur les 16 cases pour le roi noir (black King)
      for bK in cls.bK_SQUARES:
        # Boucle sur les 64 cases pour le roi blanc (white King)
        for wK in chess.SQUARES:
          if bK != wK:
            # Boucle sur les 64 cases pour le fou blanc (white Bishop)
            for wB in chess.SQUARES:
              if wB != bK and wB != wK:
                # Boucle sur les 64 cases pour le cavalier blanc (white Knight)
                for wN in chess.SQUARES:
                  if wN != bK and wN != wK and wN != wB:
                    yield Board_KBNvK(bK, wK, wB, wN)
        
                    
# =============================================================================
# Recherche des positions terminales                     
# =============================================================================
import numpy as np  # manipulation de tableaux NumPy

# Tables d'évaluation des positions pour celui dont c'est le tour de jouer :
#   -99   : position impossible (ne peut pas se rencontrer dans une partie)
#   +99   : position non évaluée
#   n = 0 : partie perdue (mat)
#   n > 0 : gain possible en n 1/2 coups (n valeur minimale)
#   n < 0 : risque de perdre en n 1/2 coups (si l'adversaire joue au mieux)
#   100   : partie nulle
bBoard_eval = np.full(16*63*62*61, +99, dtype=np.int8)
wBoard_eval = np.full(16*63*62*61, +99, dtype=np.int8)

# Boucle sur les positions canoniques pour identifier :
#  - les positions impossibles
#  - les positions où les noirs sont mat
#  - les positions où les noirs sont pat ou s'ils peuvent prendre une pièce
bBoard_loss = []  # liste de positions perdantes avec trait aux noirs
bBoard_draw = []  # liste de positions nulles avec trait aux noirs
for board in Board_KBNvK.canonical_boards():
  # index de la position et situation des pièces sur l'échiquier
  index = board.index()
  bK = board.king(chess.BLACK)
  wK = board.king(chess.WHITE)
  wB = list(board.pieces(chess.BISHOP, chess.WHITE))[0]
  wN = list(board.pieces(chess.KNIGHT, chess.WHITE))[0]

  # rois sur des cases adjacentes --> position impossible
  if chess.square_distance(bK, wK) == 1:
    bBoard_eval[index] = wBoard_eval[index] = -99
  else:
    # trait aux noirs
    board.turn = chess.BLACK
    # roi noir en échec --> position impossible si le trait était aux blancs
    if board.is_check():
      wBoard_eval[index] = -99
    # roi noir mat
    if board.is_checkmate():
      bBoard_eval[index] = 0
      bBoard_loss.append(board)
    # partie déclarée nulle : pat ou prise possible du Fou ou du Cavalier
    elif board.is_stalemate() \
          or (board.is_attacked_by(chess.BLACK, wB) 
               and not board.is_attacked_by(chess.WHITE, wB)) \
          or (board.is_attacked_by(chess.BLACK, wN) 
               and not board.is_attacked_by(chess.WHITE, wN, \
                    board.occupied^board.pieces_mask(chess.KING, chess.BLACK))):
      bBoard_eval[index] = 100
      bBoard_draw.append(board)
# Affichage des résultats
print(f"Roi échec et mat -> {len(bBoard_loss)} positions")
print(f"Roi pat ou prise possible fou/cavalier -> {len(bBoard_draw)} positions")
print()


# =============================================================================
# Analyse rétrograde pour trouver les positions avec mat forcé
# =============================================================================
nb_halfmoves = 0  # pour compter les 1/2 coups
while True:
  
  # recherche de nouvelles positions gagnantes blanches
  wBoard_win = [] # liste de positions gagnantes avec trait aux blancs
  nb_halfmoves += 1
  for board in bBoard_loss:
    # analyse des coups inverses blancs
    board.turn = chess.WHITE
    for reverse_move in board.legal_moves:
      # pas de prise du roi noir s'il est en échec
      if reverse_move.to_square != board.king(chess.BLACK):
        # validation du coup inverse
        prev_board = board.play_move(reverse_move)
        # position à enregistrer si non évaluée précédemment
        if wBoard_eval[prev_board.index()] == +99:
          wBoard_eval[prev_board.index()] = nb_halfmoves
          prev_board.turn = chess.WHITE
          wBoard_win.append(prev_board)
  # affichage du résultat
  if len(wBoard_win) == 0: break
  print(f"Blancs{len(wBoard_win):7} positions mat en{nb_halfmoves:3} ½ coups")

  # recherche de nouvelles positions perdantes noires
  bBoard_loss = [] # liste de positions perdantes avec trait aux noirs
  nb_halfmoves += 1
  for board in wBoard_win:
    # analyse des coups inverses noirs (avec possibilité d'être en échec)
    board.turn = chess.BLACK
    for reverse_move in board.pseudo_legal_moves:
      # pas de prise de pièce en coup inverse
      if not board.piece_at(reverse_move.to_square):
        # validation du coup inverse
        prev_board = board.play_move(reverse_move)
        if bBoard_eval[prev_board.index()] == +99:
          # analyse des coups possibles pour les noirs
          prev_board.turn = chess.BLACK
          for coup in prev_board.legal_moves:
            # validation du coup
            next_board = prev_board.play_move(coup)
            # arrêt si position non gagnante pour les blancs
            if not (1 <= wBoard_eval[next_board.index()] <= 90): break
          else:
            # perte forcée car tous les coups conduisent à une position gagnante
            bBoard_eval[prev_board.index()] = -nb_halfmoves
            prev_board.turn = chess.BLACK
            bBoard_loss.append(prev_board)
  # affichage du résultat
  if len(bBoard_loss) == 0: break
  print(f" Noirs{len(bBoard_loss):7} positions mat en{nb_halfmoves:3} ½ coups")
