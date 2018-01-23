import sys, numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

#0=drama,1=comedy,2=animated,3=action/adventure

def random_forest_class(raw_test_set):
    x_train=[]
    y_train=[]
    count=0
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_features = 1024)
    with open("movieTraining2.txt", "r") as mtf:
        for line in mtf:
            temp = line.split("||")            
            x_train.append(temp[4])
            y_train.append(temp[0])             
            count+=1
            '''
            if count==5:
                raw_test_set.append(temp[3])
            elif count==127:
                raw_test_set.append(temp[3])
                print(temp[1])
            elif count==157:
                raw_test_set.append(temp[3])
                print(temp[1])
            elif count==394:
                raw_test_set.append(temp[3])
            '''

    train_data_feat = vectorizer.fit_transform(x_train).toarray()
    randoforest = RandomForestClassifier(n_estimators = 100)
    randoforest = randoforest.fit(train_data_feat, y_train)        
    test_data_feat = vectorizer.transform(raw_test_set).toarray()
    result = randoforest.predict(test_data_feat)
    return result
    #print(len(y_train),len(raw_test_set))
    #print(x_train[220])

def naive_bayes_class(raw_test_set):
    x_train=[]
    y_train=[]
    count=0
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_features = 1024)
    with open("movieTraining2.txt", "r") as mtf:
        for line in mtf:
            temp = line.split("||")            
            x_train.append(temp[4])
            y_train.append(temp[1])             
            count+=1
            '''
            if count==5:
                raw_test_set.append(temp[3])
            elif count==127:
                raw_test_set.append(temp[3])
                print(temp[1])
            elif count==157:
                raw_test_set.append(temp[3])
                print(temp[1])
            elif count==394:
                raw_test_set.append(temp[3])
            '''
    
    train_data_feat = vectorizer.fit_transform(x_train).toarray()
    nb = MultinomialNB()
    nb = nb.fit(train_data_feat, y_train)
    test_data_feat = vectorizer.transform(raw_test_set).toarray()
    result = nb.predict(test_data_feat)
    return result    

def int2class(lizt, bol):
    if bol:
        for i in lizt:
            if int(i)==0:
                print('drama')
            elif int(i)==1:
                print('comedy')
            elif int(i)==2:
                print('animated')
            elif int(i)==3:
                print('action')
            elif int(i)==4:
                print('horror')
    else:
        for i in lizt:
            print(i)


rts=[
    "When Dipper and Mabel Pines get sent to their great-uncle Stan's shop in Gravity Falls, Oregon for the summer, they think it will be boring. But when Dipper find a strange journal in the woods, they learn about some strange secrets about the town. Welcome to Gravity Falls. Just north of Normal, west of Weird.",
    "When chemistry teacher Walter White is diagnosed with Stage III cancer and given only two years to live, he decides he has nothing to lose. He lives with his teenage son, who has cerebral palsy, and his wife, in New Mexico. Determined to ensure that his family will have a secure future, Walt embarks on a career of drugs and crime. He proves to be remarkably proficient in this new world as he begins manufacturing and selling methamphetamine with one of his former students. The series tracks the impacts of a fatal diagnosis on a regular, hard working man, and explores how a fatal diagnosis affects his morality and transforms him into a major player of the drug trade.",
    "A 19th century Western. Chon Wang is a clumsy Imperial Guard to the Emperor of China. When Princess Pei Pei is kidnapped from the Forbidden City, Wang feels personally responsible and insists on joining the guards sent to rescue the Princess, who has been whisked away to the United States. In Nevada and hot on the trail of the kidnappers, Wang is separated from the group and soon finds himself an unlikely partner with Roy O'Bannon, a small time robber with delusions of grandeur. Together, the two forge onto one misadventure after another.",
    "A former lawyer attends a community college when it is discovered he faked his bachelor degree. In an attempt to get with a student in his Spanish class he forms a Spanish study group. To his surprise more people attend the study group and the group of misfits form an unlikely community.",
    "A Michigan farmer and a prospector form a partnership in the California gold country. Their adventures include buying and sharing a wife, hijacking a stage, kidnaping six prostitutes, and turning their mining camp into a boomtown. Along the way there is plenty of drinking, gambling, and singing. They even find time to do some creative gold mining.",
    "Dory is a wide-eyed, blue tang fish who suffers from memory loss every 10 seconds or so. The one thing she can remember is that she somehow became separated from her parents as a child. With help from her friends Nemo and Marlin, Dory embarks on an epic adventure to find them. Her journey brings her to the Marine Life Institute, a conservatory that houses diverse ocean species.",
    "A US research station, Antarctica, early-winter 1982. The base is suddenly buzzed by a helicopter from the nearby Norwegian research station. They are trying to kill a dog that has escaped from their base. After the destruction of the Norwegian chopper the members of the US team fly to the Norwegian base, only to discover them all dead or missing. They do find the remains of a strange creature the Norwegians burned. The Americans take it to their base and deduce that it is an alien life form. After a while it is apparent that the alien can take over and assimilate into other life forms, including humans, and can spread like a virus.",
    "This is a shit movie"
    ]

#res = naive_bayes_class(rts)
#res = random_forest_class(rts)

#x = [] 
#print("Please Describe a Story:")
#x.append(input().strip())
res = naive_bayes_class(rts)
int2class(res,True)
res = random_forest_class(rts)
int2class(res,False)

    