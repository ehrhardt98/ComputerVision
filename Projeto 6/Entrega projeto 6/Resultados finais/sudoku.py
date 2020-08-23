import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import morphology
import operator
from tensorflow import keras # imports do tensorflow geram muitos warnings no ubuntu, mas nao parecem importar
from tensorflow.keras.models import load_model
from skimage.color import label2rgb
from skimage.measure import regionprops
import sys
import os
import os.path


if len(sys.argv) > 1:
    if sys.argv[1] in os.listdir():
        img_sudoku = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY)
    else:
        print('O diretorio {} nao foi encontrado'.format(sys.argv[1]))
        sys.exit()

else:
    print('\n\n')
    print('Pressione a BARRA DE ESPACO para tirar a foto do sudoku, ou ESC para sair')
    print('\n\n')
    cap = cv2.VideoCapture(0)

    _, frame = cap.read()
    print(frame.shape)
    while frame is not None:
        frame = frame[:, ::-1, :]

        cv2.imshow('frame', frame)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            print('Tchau')
            sys.exit()
        elif k == ord(' '):
            filename = f'frame.png' 
            cv2.imwrite(filename, frame)
            print(f'Saved {filename}')

            img_sudoku = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)

        _, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


sudoku_test = cv2.adaptiveThreshold(img_sudoku, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5) #transformar a imagem em binária com threshold adaptativo

# mudança de perspectiva para que o grid do sudoku fique reto e ocupe a imagem inteira.

w, h = sudoku_test.shape

contours, hierarchy = cv2.findContours(sudoku_test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if len(contours) != 0:
    c2 = max(contours, key = cv2.contourArea)

corners = []
i = 0
while len(corners) != 4:
    corners = cv2.approxPolyDP(c2, i, True)
    i += 1

for corner in corners:
    if corner[0][0] < w/2:
        if corner[0][1] < h/2:
            ul = corner[0] # upper left
        else:
            ll = corner[0] # lower left
    else:
        if corner[0][1] < h/2:
            ur = corner[0] # upper right
        else:
            lr = corner[0] # lower right

cols, rows = 2000, 2000 # redimensionar para o plot não ficar achatado

original = np.float32([ul, ll, ur, lr])
perspec = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])


M = cv2.getPerspectiveTransform(original, perspec)
grid_crop = cv2.warpPerspective(sudoku_test, M, (cols, rows))

grid_crop = cv2.rectangle(grid_crop, (1,1), (cols-1, rows-1), (255,255,255), thickness=30)

grid_crop = morphology.remove_small_objects(grid_crop, min_size=9000)

img = morphology.dilation(grid_crop, morphology.square(5)) > 0 # dilation para deixas as linhas mais largas, para que os quadrados (props) fiquem bem definidos
img1 = morphology.remove_small_objects(img, min_size=100)
img1 = morphology.remove_small_holes(img1, 110, connectivity=2)
img1 = morphology.remove_small_holes(img1, 110)

verticals = morphology.erosion(img1, morphology.rectangle(130, 1)) # separar as linhas verticais

img2 = morphology.reconstruction(verticals, img1)

img3 = morphology.dilation(img2, morphology.square(5)) > 0

img = ~img3


props = regionprops(morphology.label(img))

prop_list = []
crop_list = []

soma = 0
for i in props:
    soma += i.area

soma /= len(props)

props = [i for i in props if i.area > 0.50 * soma] # selecionar os props desejados (aqueles com área desejada).

for i in range(81):
    prop_list.append(props[i].bbox)

chunks = [prop_list[x:x+9] for x in range(0, len(prop_list), 9)] # percorrer a lista de props de 9 em 9, adicionando cada linha do sudoku em uma lista separada
for i in chunks:
    i.sort(key = operator.itemgetter(1)) # ordenação da lista pelo segundo elemento (posição da primeira coluna do prop), pois os props vêm ordenados em linhas, mas a ordem das linhas pode estar errada.
chunks = [item for sublist in chunks for item in sublist] # reagrupar as listas de props em uma só

tem_ou_nao = []
tiles = []
hist_list = []

for i in chunks: # criar os respectivos crops de cada prop, já plotando cada um em sua respectiva posição
    crop = grid_crop[i[0]:i[2], i[1]:i[3]]
    tiles.append(crop)
    soma = np.sum(grid_crop[i[0]:i[2], i[1]:i[3]])
    tem_ou_nao.append(soma/((i[2]-i[0]) * (i[3]-i[1])))


media = np.sum(tem_ou_nao) / len(tem_ou_nao)

tem_ou_nao_bool = tem_ou_nao > media # filtrar os tiles que têm números para impedir predicts desnecessários e errados.

modelo = load_model("modelo.h5")# carrega o modelo previamente treinado

prediction_list = []
for i in range(len(tem_ou_nao_bool)): # realiza o predict para cada tile que tem número
    if tem_ou_nao_bool[i]:
        tile = cv2.resize(tiles[i], (28, 28))
        tile = tile.reshape((1, 28, 28, 1))
        prediction = np.argmax(modelo.predict(tile))
        prediction_list.append(prediction)
    else:
        prediction_list.append(0)
        continue


# funcoes para resolver e plotar o tabuleiro retiradas de uma thread do stack overflow

def solve(bo):
    find = find_empty(bo)
    if not find:
        return bo
    else:
        row, col = find

    for i in range(1,10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i

            if solve(bo):
                return True

            bo[row][col] = 0

    return False


def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False

    return True


def print_board(bo):
    print("")
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("-------+--------+-------")

        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                if bo[i][j] == 0:
                    print("-")
                else:
                    print(bo[i][j])
            else:
                if bo[i][j] == 0:
                    print("-" + " ", end="")
                else: 
                    print(str(bo[i][j]) + " ", end="")
    print("")


def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)

    return None

board = list(np.array(prediction_list).reshape(9,9))

board_solve = board.copy()
solved = solve(board_solve)


if not solved:
    print('O sudoku utilizado nao tem solucao.')
    sys.exit()



w, h = sudoku_test.shape

contours, hierarchy = cv2.findContours(sudoku_test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if len(contours) != 0:
    c2 = max(contours, key = cv2.contourArea)

corners = []
i = 0
while len(corners) != 4:
    corners = cv2.approxPolyDP(c2, i, True)
    i += 1

for corner in corners:
    if corner[0][0] < w/2:
        if corner[0][1] < h/2:
            ul = corner[0] # upper left
        else:
            ll = corner[0] # lower left
    else:
        if corner[0][1] < h/2:
            ur = corner[0] # upper right
        else:
            lr = corner[0] # lower right

cols, rows = 2000, 2000 # redimensionar para o plot não ficar achatado

original = np.float32([ul, ll, ur, lr])
perspec = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])


M = cv2.getPerspectiveTransform(original, perspec)
grid_crop = cv2.warpPerspective(img_sudoku, M, (cols, rows))

grid_crop = cv2.rectangle(grid_crop, (1,1), (cols-1, rows-1), (255,255,255), thickness=30)

grid_crop = morphology.remove_small_objects(grid_crop, min_size=9000)

board = list(np.array(prediction_list).reshape(9,9)) # refaz o board, pois a funcao solve() resolve o board criado acima

# codigo para printar os resultados sobre a imagem original

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale  = 4
fontColor = (0,0,0)
lineType = 2
thickness = 10

img_solve = (grid_crop.astype("uint8"))

img_solve = cv2.cvtColor(img_solve, cv2.COLOR_BGR2RGB)

for i in range(9):
    for j in range(9):
        if board[i][j] == 0:
            min_row, min_col, max_row, max_col = chunks[i*9 + j]
            row_wr = int(min_row + 2000/15)
            col_wr = int(max_col - 2000/15)
            cv2.putText(img_solve, str(board_solve[i][j]), (col_wr, row_wr), font, fontScale, fontColor, thickness, lineType)


plt.figure(figsize=(12,12))
plt.imshow(img_solve)
plt.show()