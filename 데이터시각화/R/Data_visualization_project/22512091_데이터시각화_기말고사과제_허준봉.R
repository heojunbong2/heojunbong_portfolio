
set.seed(42)

                 /* 민주당 자료 생성 */
  
a<-sample(x=20:30, size=10,replace=TRUE)
a_0.1mi<-sample(x=20:30, size=50,replace=TRUE)
a_2plus<-sample(x=30:40, size=50,replace=TRUE)

a_30401minus<-sample(x=30:40, size=55,replace=TRUE)
a_30402minus<-sample(x=30:40, size=55,replace=TRUE)

a_30403minus<-sample(x=30:40, size=35,replace=TRUE)
a_30405minus<-sample(x=30:40, size=30,replace=TRUE)
a_304010minus<-sample(x=30:40, size=50,replace=TRUE)


a_1plus<-sample(x=40:55, size=30,replace=TRUE)
a_2_2plus<-sample(x=40:55, size=30,replace=TRUE)
a_5plus<-sample(x=40:55, size=45,replace=TRUE)

a_40555minus<-sample(x=40:55,size=50,replace=TRUE)
a_40552minus<-sample(x=40:55,size=50,replace=TRUE)
a_40553minus<-sample(x=40:55,size=50,replace=TRUE)
a_40551minus<-sample(x=40:55,size=50,replace=TRUE)

a_40553minus<-sample(x=40:55,size=10,replace=TRUE)
a_40554minus<-sample(x=40:55,size=10,replace=TRUE)
b_0.1<-a+1
b_0.1mi<-a_0.1mi-1
b_2plus<-a_2plus+2

b_30401minus<-a_30401minus-1
b_30402minus<-a_30402minus-2

b_30403minus<-a_30403minus-3
b_30405minus<-a_30405minus-5
b_304010minus<-a_304010minus-4

b_1plus<-a_1plus+1
b_2_2plus<-a_2_2plus+2
b_5plus<-a_5plus+5
b_40555minus<-a_40555minus-5
b_40551minus<-a_40551minus-1

b_40552minus<-a_40552minus-2

b_40553minus<-a_40553minus-3

b_40554minus<-a_40554minus-4

data_republic<-data.frame(data_X=a,data_y=b_0.1)
data_republic_2plus<-data.frame(data_X=a_2plus,data_y=b_2plus)

data_republic_30401minus<-data.frame(data_X=a_30401minus,data_y=b_30401minus)
data_republic_30402minus<-data.frame(data_X=a_30402minus,data_y=b_30402minus)


data_republic_30403minus<-data.frame(data_X=a_30403minus,data_y=b_30403minus)
data_republic_30405minus<-data.frame(data_X=a_30405minus,data_y=b_30405minus)
data_republic_304010minus<-data.frame(data_X=a_304010minus,data_y=b_304010minus)
data_republic_1plus<-data.frame(data_X=a_1plus,data_y=b_1plus)
data_republic_2_2plus<-data.frame(data_X=a_2_2plus,data_y=b_2_2plus)
data_republic_5plus<-data.frame(data_X=a_5plus,data_y=b_5plus)
data_republic_40555minus<-data.frame(data_X=a_40555minus,data_y=b_40555minus)
data_republic_40552minus<-data.frame(data_X=a_40552minus,data_y=b_40552minus)
data_republic_40553minus<-data.frame(data_X=a_40553minus,data_y=b_40553minus)
data_republic_40554minus<-data.frame(data_X=a_40554minus,data_y=b_40554minus)


data_republic_mi<-data.frame(data_X=a_0.1mi,data_y=b_0.1mi)
data_republic$label='민주당'
data_republic_mi$label='민주당'
data_republic_2plus$label='민주당'
data_republic_2_2plus$label='민주당'
data_republic_1plus$label='민주당'
data_republic_5plus$label='민주당'

data_republic_30401minus$label='민주당'
data_republic_30402minus$label='민주당'

data_republic_30403minus$label='민주당'
data_republic_30405minus$label='민주당'
data_republic_304010minus$label='민주당'
data_republic_40555minus$label='민주당'
data_republic_40552minus$label='민주당'
data_republic_40553minus$label='민주당'
data_republic_40554minus$label='민주당'
#data_republic_40551minus$label='민주당'

data1<-rbind(data_republic, data_republic_mi,data_republic_2plus,data_republic_2_2plus,data_republic_1plus,
            data_republic_5plus,data_republic_30401minus,data_republic_30402minus,data_republic_30403minus,
            data_republic_30405minus,data_republic_304010minus,data_republic_40555minus,
            data_republic_40552minus,data_republic_40553minus,data_republic_40554minus)


                       /* 국민의당 자료 생성 */

c<-sample(x=80:90, size=3,replace=TRUE)
c_0.1mi<-sample(x=40:55, size=55,replace=TRUE)
c_2plus<-sample(x=55:65, size=45,replace=TRUE)
c_5plus<-sample(x=55:65, size=45,replace=TRUE)
c_3plus<-sample(x=55:65, size=55,replace=TRUE)
c_1plus<-sample(x=55:65, size=55,replace=TRUE)


c_0.3plus<-sample(x=65:80, size=35,replace=TRUE)
c_60703plus<-sample(x=65:70, size=45,replace=TRUE)
c_60704plus<-sample(x=65:70, size=45,replace=TRUE)
c_60705plus<-sample(x=65:70, size=45,replace=TRUE)
c_60703minus<-sample(x=65:70, size=45,replace=TRUE)
c_60705minus<-sample(x=65:70, size=45,replace=TRUE)
c_607010minus<-sample(x=65:70, size=50,replace=TRUE)

c_40555minus<-sample(x=45:55, size=30,replace=TRUE)
c_40552minus<-sample(x=45:55, size=40,replace=TRUE)
c_40553minus<-sample(x=45:55, size=30,replace=TRUE)
c_40554minus<-sample(x=45:55, size=30,replace=TRUE)
c_55655minus<-sample(x=55:65, size=30,replace=TRUE)
c_55654minus<-sample(x=55:65, size=30,replace=TRUE)
c_55653minus<-sample(x=55:65, size=30,replace=TRUE)
c_55652minus<-sample(x=55:65, size=40,replace=TRUE)
c_55651minus<-sample(x=55:65, size=40,replace=TRUE)



#a_5plus<-sample(x=40:55, size=10,replace=TRUE)

d_0.1<-c+0.1
d_0.1mi<-c_0.1mi-0.1
d_2plus<-c_2plus+2
d_5plus<-c_5plus+5
d_3plus<-c_3plus+3
d_1plus<-c_1plus+1



d_0.3plus<-c_0.3plus+0.3
d_60703plus<-c_60703plus+3
d_60704plus<-c_60704plus+4
d_60705plus<-c_60705plus+5

d_55655minus<-c_55655minus-5
d_55654minus<-c_55654minus-4
d_55653minus<-c_55653minus-3
d_55652minus<-c_55652minus-2
d_55651minus<-c_55651minus-1


d_60703minus<-c_60703minus-3
d_60705minus<-c_60705minus-5
d_607010minus<-c_607010minus-4

d_40555minus<-c_40555minus-4.5
d_40552minus<-c_40552minus-2
d_40553minus<-c_40553minus-3
d_40554minus<-c_40554minus-4

#b_5plus<-a_5plus+5


data_power<-data.frame(data_X=c,data_y=d_0.1)
data_power_2plus<-data.frame(data_X=c_2plus,data_y=d_2plus)
data_power_5plus<-data.frame(data_X=c_5plus,data_y=d_5plus)
data_power_3plus<-data.frame(data_X=c_3plus,data_y=d_3plus)
data_power_1plus<-data.frame(data_X=c_1plus,data_y=d_1plus)


data_power_0.3plus<-data.frame(data_X=c_0.3plus,data_y=d_0.3plus)
data_power_60703plus<-data.frame(data_X=c_60703plus,data_y=d_60703plus)
data_power_60704plus<-data.frame(data_X=c_60704plus,data_y=d_60704plus)

data_power_60705plus<-data.frame(data_X=c_60705plus,data_y=d_60705plus)

data_power_40555minus<-data.frame(data_X=c_40555minus,data_y=d_40555minus)
data_power_40552minus<-data.frame(data_X=c_40552minus,data_y=d_40552minus)

data_power_40553minus<-data.frame(data_X=c_40553minus,data_y=d_40553minus)

data_power_40554minus<-data.frame(data_X=c_40554minus,data_y=d_40554minus)

data_power_55655minus<-data.frame(data_X=c_55655minus,data_y=d_55655minus)
data_power_55654minus<-data.frame(data_X=c_55654minus,data_y=d_55654minus)
data_power_55653minus<-data.frame(data_X=c_55653minus,data_y=d_55653minus)

data_power_55652minus<-data.frame(data_X=c_55652minus,data_y=d_55652minus)
data_power_55651minus<-data.frame(data_X=c_55651minus,data_y=d_55651minus)


data_power_60703minus<-data.frame(data_X=c_60703minus,data_y=d_60703minus)
data_power_60705minus<-data.frame(data_X=c_60705minus,data_y=d_60705minus)
data_power_607010minus<-data.frame(data_X=c_607010minus,data_y=d_607010minus)

data_power_55651minus
#data_republic_5plus<-data.frame(data_X=a_5plus,data_y=b_5plus)

data_power_mi<-data.frame(data_X=c_0.1mi,data_y=d_0.1mi)

data_power$label='국민의 힘'
data_power_mi$label='국민의 힘'
data_power_2plus$label='국민의 힘'
data_power_5plus$label='국민의 힘'
data_power_3plus$label='국민의 힘'
data_power_1plus$label='국민의 힘'

data_power_0.3plus$label='국민의 힘'

data_power_60703plus$label='국민의 힘'
data_power_60704plus$label='국민의 힘'
data_power_60705plus$label='국민의 힘'

data_power_55655minus$label='국민의 힘'
data_power_55654minus$label='국민의 힘'
data_power_55653minus$label='국민의 힘'
data_power_55652minus$label='국민의 힘'
data_power_55651minus$label='국민의 힘'


data_power_60703minus$label='국민의 힘'
data_power_60705minus$label='국민의 힘'
data_power_607010minus$label='국민의 힘'
data_power_40555minus$label='국민의 힘'
data_power_40552minus$label='국민의 힘'
data_power_40553minus$label='국민의 힘'
data_power_40554minus$label='국민의 힘'


data2<-rbind(data_power, data_power_mi,data_power_1plus,data_power_2plus,data_power_5plus,
             data_power_3plus,data_power_0.3plus,data_power_60703plus,
             data_power_60704plus,data_power_60705plus,data_power_60703minus,
             data_power_60705minus,data_power_607010minus,data_power_40555minus,
             data_power_40552minus,data_power_40553minus,data_power_40554minus,
             data_power_55655minus,data_power_55654minus,data_power_55653minus,
             data_power_55652minus,data_power_55651minus)

data<-rbind(data1,data2)

                      /* 그림 그리기 */ 

library(dplyr)
label<-data %>% summarise(
  data_X=10,
  data_y=80,
  label="실제 득표율이 올라갈수록 예측 득표율이 \n 증가하는 경향이 있다.")

class_avg<-data %>% group_by(label) %>%
  summarise(data_X=median(data_X),
            data_y=median(data_y))

install.packages("ggrepl")

library(ggrepel)
library(ggplot2)
fig<-ggplot(data,aes(x=data_X,y=data_y))+geom_point(aes(x=data_X,y=data_y,colour=factor(label)),alpha=0.3,position='jitter',size=3)+
  scale_colour_manual(values=c("red","blue"))+
  coord_fixed(0.75)+
  scale_x_continuous(breaks=seq(10,100, by=10),labels=paste0(seq(10,100,by=10),"%"))+
  scale_y_continuous(breaks=seq(0,100, by=20), labels=paste0(seq(0,100,by=20),"%"))+
  labs(title="실제득표율(%)과 예측 득표율(%) 상관관계 분석",
       subtitle="<실제득표율과 예측 득표율은 매우 강한 양의 상관관계를 가지고 있다>"
       ,x='< 실제 득표율(%) >',y='< 예측 득표율(%) >',
       color='정당 이름')
fig2<-fig + theme_bw()+
  theme(axis.text.x=element_text(size=10,face='bold',color='black'),
    axis.text.y =element_text(size=10,face='bold',color='black'),
    title=element_text(size=20, face='bold'))+
  theme(plot.title=element_text(hjust=0.5),
        plot.subtitle=element_text(hjust=0.5,size=15))+
  geom_text(
    data=label,
    mapping=aes(x=data_X,y=data_y,label=label),
    vjust='top',
    hjust='left')

fig3<-fig2 + ggrepel::geom_label_repel(data=class_avg,
    mapping=aes(label=label, color=label),
    size=3,
    label.size=0)+
  theme(legend.position='none')


fig3



