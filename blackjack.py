import pygame
import sys
import random
import copy
import time
from pygame.locals import *
from cards import *


# Generate and remove an card from cList and append it to xList.
# Return the card, and whether the card is an Ace
def genCard(cList, xList):
    cA = 0
    card = random.choice(cList)
    cList.remove(card)
    xList.append(card)
    if card in cardA:
        cA = 1
    return card, cA


def initGame(cList, uList, dList):
    # Generates two cards for dealer and user, one at a time for each.
    # Returns if card is Ace and the total amount of the cards per person.
    userA = 0
    dealA = 0
    card1, cA = genCard(cList, uList)
    userA += cA
    card2, cA = genCard(cList, dList)
    dealA += cA
    dealAFirst = copy.deepcopy(dealA)
    card3, cA = genCard(cList, uList)
    userA += cA
    card4, cA = genCard(cList, dList)
    dealA += cA
    # The values are explained below when calling the function
    return getAmt(card1) + getAmt(card3), userA, getAmt(card2) + \
        getAmt(card4), dealA, getAmt(card2), dealAFirst


def make_state(userSum, userA, dealFirst, dealAFirst):
    # Eliminate duplicated bust cases
    if userSum > 21:
        userSum = 22
    # userSum: sum of user's cards
    # userA: number of user's Aces
    # dealFirst: value of dealer's first card
    # dealAFirst: whether dealer's first card is Ace
    return (userSum, min(userA, 4), dealFirst, dealAFirst)


def main():
    ccards = copy.copy(cards)
    stand = False
    userCard = []
    dealCard = []
    winNum = 0
    loseNum = 0
    # Initialize Game
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption('Blackjack')
    font = pygame.font.SysFont("", 20)
    hitTxt = font.render('Hit', 1, black)
    standTxt = font.render('Stand', 1, black)
    restartTxt = font.render('Restart', 1, black)
    MCTxt = font.render('MC', 1, blue)
    TDTxt = font.render('TD', 1, blue)
    QLTxt = font.render('QL', 1, blue)
    gameoverTxt = font.render('End of Round', 1, white)
    # Prepare table of utilities
    MCvalues = {}
    TDvalues = {}
    Qvalues = {}
    MC_G = dict()
    TD_G = dict()
    Q_G = dict()
    alpha = 0.01
    gamma = 0.9
    # i iterates through the sum of user's cards. It is set to 22 if the user went bust.
    # j iterates through the value of the dealer's first card. Ace is eleven.
    # a1 is the number of Aces that the user has.
    # a2 denotes whether the dealer's first card is Ace.
    for i in range(2, 23):
        for j in range(2, 12):
            for a1 in range(0, 5):
                for a2 in range(0, 2):
                    s = (i, a1, j, a2)
                    # utility computed by MC-learning
                    MCvalues[s] = 0
                    MC_G[s] = []
                    # utility computed by TD-learning
                    TDvalues[s] = 0
                    TD_G[s] = []
                    # first element is Q value of "Hit", second element is Q
                    # value of "Stand"
                    if i == 22:
                        Qvalues[s] = [-1, -1]
                    Qvalues[s] = [0,0]
    states = [k for k in MCvalues.keys()]
    # userSum: sum of user's cards
    # userA: number of user's Aces
    # dealSum: sum of dealer's cards (including hidden one)
    # dealA: number of all dealer's Aces,
    # dealFirst: value of dealer's first card
    # dealAFirst: whether dealer's first card is Ace
    userSum, userA, dealSum, dealA, dealFirst, dealAFirst = initGame(
        ccards, userCard, dealCard)
    state = make_state(userSum, userA, dealFirst, dealAFirst)
    # Fill background
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((80, 150, 15))
    hitB = pygame.draw.rect(background, gray, (10, 445, 75, 25))
    standB = pygame.draw.rect(background, gray, (95, 445, 75, 25))
    MCB = pygame.draw.rect(background, white, (180, 445, 75, 25))
    TDB = pygame.draw.rect(background, white, (265, 445, 75, 25))
    QLB = pygame.draw.rect(background, white, (350, 445, 75, 25))
    autoMC = False
    autoTD = False
    autoQL = False
    # Event loop
    while True:
        # Our state information does not take into account of number of cards
        # So it's ok to ignore the rule of winning if getting 5 cards without
        # going bust
        c1 = (copy.copy(ccards), copy.copy(userCard))
        c2 = (copy.copy(ccards), copy.copy(dealCard))
        if (userSum >= 21 and userA == 0) or len(userCard) == 5:
            gameover = True
        else:
            gameover = False
        if len(userCard) == 2 and userSum == 21:
            gameover = True
        if autoMC:
            count = 0
            while count < 1000:
                count += 1
                s = random.choice(states)
                #s = state
                episode = simulate_sequence(c1, s, None)
                rewards = evaluate_episode(c2, episode)
                r = len(episode)-1
                for s in episode:
                    MC_G[s].append(gamma ** r * rewards)
                    MCvalues[s] = sum(MC_G[s])/len(MC_G[s])
                    r -= 1
            # MC Learning (erase the dummy +1 of course)
            # Compute the utilities of all states under the policy "Always hit
            # if below 17"
        if autoTD:
            # TD Learning (erase the dummy +1 of course)
            # Compute the utilities of all states under the policy "Always hit
            # if below 17"
            count = 0
            while count < 1000:
                count += 1
                s = random.choice(states)
                #s = state
                while s is not None:
                    c2 = (copy.copy(ccards), copy.copy(dealCard))
                    rewards = evaluate_episode(c2, [s])
                    next_s = simulate_one_step(c1, s, None)
                    if next_s is None:
                        TDvalues[s] += alpha*(rewards-TDvalues[s])
                        break
                    TDvalues[s] = TDvalues[s]+alpha*(rewards+gamma*TDvalues[next_s]-TDvalues[s])
                    s = next_s
        if autoQL:
            # Q-Learning (erase the dummy +1 of course)
            # For each state, compute the Q value of the action "Hit" and
            # "Stand"
            # 0 for stand, 1 for hit
            Q = Qvalues
            #s = random.choice(states)
            s = state
            eps = 0.25
            while s is not None:
                a = pick_action(s, eps, Q)
                next_s = simulate_one_step(c1, (s,a), None, q=True)
                rewards = evaluate_episode(c2, [s], q=True)
                # when user stands
                if next_s is None:
                    Q[s][a] += alpha*(rewards-Q[s][a])
                    break
                # when next_bust
                if next_s[0] == 22:
                    # hit, but next_s bust
                    if a == 1:
                        rewards = evaluate_episode(c2, [next_s], q=True)
                        rewards = -1
                        Q[s][a] += alpha*(rewards-Q[s][a])
                    # stand, but next_s bust
                    else:
                        Q[s][a] += alpha*(rewards-Q[s][a])
                    break
                if s == state:
                    m = max(Q[next_s][0]-Q[s][0],Q[next_s][1]-Q[s][1])
                Q[s][a] += alpha * (gamma*max(Q[next_s][0]-Q[s][a],Q[next_s][1]-Q[s][a]))
                s = next_s

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            # Clicking the white buttons can start or pause the learning
            # processes
            elif event.type == pygame.MOUSEBUTTONDOWN and MCB.collidepoint(pygame.mouse.get_pos()):
                autoMC = not autoMC
            elif event.type == pygame.MOUSEBUTTONDOWN and TDB.collidepoint(pygame.mouse.get_pos()):
                autoTD = not autoTD
            elif event.type == pygame.MOUSEBUTTONDOWN and QLB.collidepoint(pygame.mouse.get_pos()):
                autoQL = not autoQL
            elif event.type == pygame.MOUSEBUTTONDOWN and (gameover or stand):
                # restarts the game, updating scores
                if userSum == dealSum:
                    pass
                elif userSum <= 21 and len(userCard) == 5:
                    winNum += 1
                elif userSum <= 21 and dealSum < userSum or dealSum > 21:
                    winNum += 1
                else:
                    loseNum += 1
                gameover = False
                stand = False
                userCard = []
                dealCard = []
                ccards = copy.copy(cards)
                userSum, userA, dealSum, dealA, dealFirst, dealAFirst = initGame(
                    ccards, userCard, dealCard)
            elif event.type == pygame.MOUSEBUTTONDOWN and not (gameover or stand) and hitB.collidepoint(pygame.mouse.get_pos()):
                # Give player a card
                card, cA = genCard(ccards, userCard)
                userA += cA
                userSum += getAmt(card)
                while userSum > 21 and userA > 0:
                    userA -= 1
                    userSum -= 10
            elif event.type == pygame.MOUSEBUTTONDOWN and not gameover and standB.collidepoint(pygame.mouse.get_pos()):
                # Dealer plays, user stands
                stand = True
                if dealSum == 21:
                    pass
                else:
                    while dealSum <= userSum and dealSum < 17:
                        card, cA = genCard(ccards, dealCard)
                        dealA += cA
                        dealSum += getAmt(card)
                        while dealSum > 21 and dealA > 0:
                            dealA -= 1
                            dealSum -= 10
        state = make_state(userSum, userA, dealFirst, dealAFirst)
        MCU = font.render(
            'MC-Utility of Current State: %f' %
            MCvalues[state], 1, black)
        TDU = font.render(
            'TD-Utility of Current State: %f' %
            TDvalues[state], 1, black)
        QV = font.render('Q values: (Hit) %f (Stand) %f' %
                         (Qvalues[state][1],Qvalues[state][0]), 1, black)
        winTxt = font.render('Wins: %i' % winNum, 1, white)
        loseTxt = font.render('Losses: %i' % loseNum, 1, white)
        screen.blit(background, (0, 0))
        screen.blit(hitTxt, (39, 448))
        screen.blit(standTxt, (116, 448))
        screen.blit(MCTxt, (193, 448))
        screen.blit(TDTxt, (280, 448))
        screen.blit(QLTxt, (357, 448))
        screen.blit(winTxt, (550, 423))
        screen.blit(loseTxt, (550, 448))
        screen.blit(MCU, (20, 200))
        screen.blit(TDU, (20, 220))
        screen.blit(QV, (20, 240))
        for card in dealCard:
            x = 10 + dealCard.index(card) * 110
            screen.blit(card, (x, 10))
        screen.blit(cBack, (120, 10))
        for card in userCard:
            x = 10 + userCard.index(card) * 110
            screen.blit(card, (x, 295))
        if gameover or stand:
            screen.blit(gameoverTxt, (270, 200))
            screen.blit(dealCard[1], (120, 10))
        pygame.display.update()


def reward_to_go(s):
    userSum, userA, dealFirst, dealAFirst = s
    if userSum == 21:
        return 1
    if userSum >= 22:
        return -1


def simulate_sequence(c, state, policy):
    s = [state]
    while True:
        cList = copy.copy(c[0])
        xList = copy.copy(c[1])
        userSum, userA, dealFirst, dealAFirst = s[-1]
        if userSum > 17:
            break
        s += [make_state(userSum+getAmt(genCard(cList, xList)[0]),
              userA+genCard(cList, xList)[1], dealFirst, dealAFirst)]
    return s


def evaluate_episode(c, episode, q=False):
    s = episode[-1]
    cList = copy.copy(c[0])
    xList = copy.copy(c[1])
    userSum, userA, dealFirst, dealAFirst = s
    if userSum < 17 and q == False:
        return 0
    if userSum > 21:
        return -1
    if userSum == 21:
        return 1
    dealA = dealAFirst
    if q == True:
        dealSum = dealFirst
        cList += [xList[1]]
        xList.remove(xList[1])
    else:
        dealSum = dealFirst + getAmt(xList[1])
        if xList[1] in cardA:
            dealA = dealAFirst + 1            
    if dealSum == 21:
        return -1
    while dealSum <= userSum and dealSum < 17:
        card, cA = genCard(cList, xList)
        dealA += cA
        dealSum += getAmt(card)
        while dealSum > 21 and dealA > 0:
            dealA -= 1
            dealSum -= 10
    if dealSum > 21:
        return 1
    if dealSum > userSum:
        return -1
    else:
        return 1

# make another simulation for s
def simulate_one_step(c, s, policy, q=False):
    if q == True:
        s1, a = s
        if a == 0:
            return None
        userSum, userA, dealFirst, dealAFirst = s1
    else:
        userSum, userA, dealFirst, dealAFirst = s
    cList = copy.copy(c[0])
    xList = copy.copy(c[1])
    if userSum > 17 and q == False:
        return None
    s = make_state(userSum+getAmt(genCard(cList, xList)[0]),
                   userA+genCard(cList, xList)[1], dealFirst, dealAFirst)
    return s


def pick_action(s, eps, Q):
    if random.random() < eps:
        return random.choice([0,1])
    else:
        if Q[s][0] > Q[s][1]:
            return 0
        else:
            return 1


if __name__ == '__main__':
    main()
