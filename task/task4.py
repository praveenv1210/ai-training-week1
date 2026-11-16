import numpy as np

marks =np.array([[70,80,90],
                [60,75,85],
                [88,92,79],
                [55,65,70]])

Shape=np.shape(marks)
print("Shape:",Shape)
weigth=np.array([0.3,0.3,0.4])
weigth_score=np.average(marks,axis=1,weights=weigth)
print("student averages:",np.round(marks.mean(axis=1),2))

print("subject average:",marks.mean(axis=0))

print("weighted scores:",np.round(weigth_score,2))