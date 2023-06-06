###############################################################################

#Title: Comparing episodes of Volcanic Unrest at Campi Flegrei Caldera, Italy.
#Author details: Mungyeom Kim, Contact details: mungyeom.kim.20@ucl.ac.uk 
#Date: 1 March 2021.
#Script description: This script performs analyses of temporal trends in monitoring data from Campi Flegrei.
# Data information: This study uses deformation, seismicity and geochemical data collected by the Vesuvius Observatory and received in a package from the UCL Hazard Centre.

###############################################################################

#Install packages ----
install.packages('tidyverse')
install.packages('gridExtra')
install.packages('ggplot2')
install.packages('dplyr')
install.packages('stats')
install.packages('lubridate')
install.packages('plotly')
install.packages('patchwork')
install.packages('dygraphs')
install.packages('ggmap')
install.packages('latticeExtra')

###############################################################################

#Load packages
library(tidyverse)
library(gridExtra)
library(ggplot2)
library(dplyr)
library(stats)
library(lubridate)
library(plotly)
library(patchwork)
library(dygraphs)
library(xts)
library(ggmap)
library(latticeExtra)

###############################################################################

#Read CSVs----
CF_Gas_BG<- read_csv('CF_Gas_BG.csv')
CF_Gas_BN<- read_csv('CF_Gas_BN.csv')
CF_Gas_Pisc<- read_csv('CF_Gas_Pisc.csv')
CF_VT_DEF<- read_csv('CF_VT_DEF.csv')

###############################################################################

#Organize the data----

##Mean VT and deformation rates
###Mean VT of each year
VT_1982 <-CF_VT_DEF[c(1:12),c(1,2)]
VT_1982$mean <- mean(VT_1982$VT)

VT_1983 <-CF_VT_DEF[c(13:24),c(1,2)]
VT_1983$mean <- mean(VT_1983$VT)

VT_1984 <-CF_VT_DEF[c(25:36),c(1,2)]
VT_1984$mean <- mean(VT_1984$VT)

VT_1985 <-CF_VT_DEF[c(37:48),c(1,2)]
VT_1985$mean <- mean(VT_1985$VT)

VT_1986 <-CF_VT_DEF[c(49:60),c(1,2)]
VT_1986$mean <- mean(VT_1986$VT)

VT_1987 <-CF_VT_DEF[c(61:72),c(1,2)]
VT_1987$mean <- mean(VT_1987$VT)

VT_1988 <-CF_VT_DEF[c(73:84),c(1,2)]
VT_1988$mean <- mean(VT_1988$VT)

VT_1989 <-CF_VT_DEF[c(85:96),c(1,2)]
VT_1989$mean <- mean(VT_1989$VT)

VT_1990 <-CF_VT_DEF[c(97:108),c(1,2)]
VT_1990$mean <- mean(VT_1990$VT)

VT_1991 <-CF_VT_DEF[c(109:120),c(1,2)]
VT_1991$mean <- mean(VT_1991$VT)

VT_1992 <-CF_VT_DEF[c(121:132),c(1,2)]
VT_1992$mean <- mean(VT_1992$VT)

VT_1993 <-CF_VT_DEF[c(133:144),c(1,2)]
VT_1993$mean <- mean(VT_1993$VT)

VT_1994 <-CF_VT_DEF[c(145:156),c(1,2)]
VT_1994$mean <- mean(VT_1994$VT)

VT_1995 <-CF_VT_DEF[c(157:168),c(1,2)]
VT_1995$mean <- mean(VT_1995$VT)

VT_1996 <-CF_VT_DEF[c(169:180),c(1,2)]
VT_1996$mean <- mean(VT_1996$VT)

VT_1997 <-CF_VT_DEF[c(181:192),c(1,2)]
VT_1997$mean <- mean(VT_1997$VT)

VT_1998 <-CF_VT_DEF[c(193:204),c(1,2)]
VT_1998$mean <- mean(VT_1998$VT)

VT_1999 <-CF_VT_DEF[c(205:216),c(1,2)]
VT_1999$mean <- mean(VT_1999$VT)

VT_2000 <-CF_VT_DEF[c(217:228),c(1,2)]
VT_2000$mean <- mean(VT_2000$VT)

VT_2001 <-CF_VT_DEF[c(229:240),c(1,2)]
VT_2001$mean <- mean(VT_2001$VT)

VT_2002 <-CF_VT_DEF[c(241:252),c(1,2)]
VT_2002$mean <- mean(VT_2002$VT)

VT_2003 <-CF_VT_DEF[c(253:264),c(1,2)]
VT_2003$mean <- mean(VT_2003$VT)

VT_2004 <-CF_VT_DEF[c(265:276),c(1,2)]
VT_2004$mean <- mean(VT_2004$VT)

VT_2005 <-CF_VT_DEF[c(277:288),c(1,2)]
VT_2005$mean <- mean(VT_2005$VT)

VT_2006 <-CF_VT_DEF[c(289:300),c(1,2)]
VT_2006$mean <- mean(VT_2006$VT)

VT_2007 <-CF_VT_DEF[c(301:312),c(1,2)]
VT_2007$mean <- mean(VT_2007$VT)

VT_2008 <-CF_VT_DEF[c(313:324),c(1,2)]
VT_2008$mean <- mean(VT_2008$VT)

VT_2009 <-CF_VT_DEF[c(325:336),c(1,2)]
VT_2009$mean <- mean(VT_2009$VT)

VT_2010 <-CF_VT_DEF[c(337:348),c(1,2)]
VT_2010$mean <- mean(VT_2010$VT)

VT_2011 <-CF_VT_DEF[c(349:360),c(1,2)]
VT_2011$mean <- mean(VT_2011$VT)

VT_2012 <-CF_VT_DEF[c(361:372),c(1,2)]
VT_2012$mean <- mean(VT_2012$VT)

VT_2013 <-CF_VT_DEF[c(373:384),c(1,2)]
VT_2013$mean <- mean(VT_2013$VT)

VT_2014 <-CF_VT_DEF[c(385:396),c(1,2)]
VT_2014$mean <- mean(VT_2014$VT)

VT_2015 <-CF_VT_DEF[c(397:408),c(1,2)]
VT_2015$mean <- mean(VT_2015$VT)

VT_2016 <-CF_VT_DEF[c(409:415),c(1,2)]
VT_2016$mean <- mean(VT_2016$VT)

#combine the data of VT
CF_VT <- bind_rows(VT_1982,VT_1983,VT_1984,VT_1985,VT_1986,VT_1987,
                   VT_1988,VT_1989,VT_1990,VT_1991,VT_1992,VT_1993,
                   VT_1994,VT_1995,VT_1996,VT_1997,VT_1998,VT_1999,
                   VT_2000,VT_2001,VT_2002,VT_2003,VT_2004,VT_2005,
                   VT_2006,VT_2007,VT_2008,VT_2009,VT_2010,VT_2011,
                   VT_2012,VT_2013,VT_2014,VT_2015,VT_2016)

###Mean DEF of each year
DEF_1982 <-CF_VT_DEF[c(1:12),c(1,3)]
DEF_1982$mean <- mean(DEF_1982$DEF, na.rm = T)

DEF_1983 <-CF_VT_DEF[c(13:24),c(1,3)]
DEF_1983$mean <- mean(DEF_1983$DEF, na.rm = T)

DEF_1984 <-CF_VT_DEF[c(25:36),c(1,3)]
DEF_1984$mean <- mean(DEF_1984$DEF, na.rm = T)

DEF_1985 <-CF_VT_DEF[c(37:48),c(1,3)]
DEF_1985$mean <- mean(DEF_1985$DEF, na.rm = T)

DEF_1986 <-CF_VT_DEF[c(49:60),c(1,3)]
DEF_1986$mean <- mean(DEF_1986$DEF, na.rm = T)

DEF_1987 <-CF_VT_DEF[c(61:72),c(1,3)]
DEF_1987$mean <- mean(DEF_1987$DEF, na.rm = T)

DEF_1988 <-CF_VT_DEF[c(73:84),c(1,3)]
DEF_1988$mean <- mean(DEF_1988$DEF, na.rm = T)

DEF_1989 <-CF_VT_DEF[c(85:96),c(1,3)]
DEF_1989$mean <- mean(DEF_1989$DEF, na.rm = T)

DEF_1990 <-CF_VT_DEF[c(97:108),c(1,3)]
DEF_1990$mean <- mean(DEF_1990$DEF, na.rm = T)

DEF_1991 <-CF_VT_DEF[c(109:120),c(1,3)]
DEF_1991$mean <- mean(DEF_1991$DEF, na.rm = T)

DEF_1992 <-CF_VT_DEF[c(121:132),c(1,3)]
DEF_1992$mean <- mean(DEF_1992$DEF, na.rm = T)

DEF_1993 <-CF_VT_DEF[c(133:144),c(1,3)]
DEF_1993$mean <- mean(DEF_1993$DEF, na.rm = T)

DEF_1994 <-CF_VT_DEF[c(145:156),c(1,3)]
DEF_1994$mean <- mean(DEF_1994$DEF, na.rm = T)

DEF_1995 <-CF_VT_DEF[c(157:168),c(1,3)]
DEF_1995$mean <- mean(DEF_1995$DEF, na.rm = T)

DEF_1996 <-CF_VT_DEF[c(169:180),c(1,3)]
DEF_1996$mean <- mean(DEF_1996$DEF, na.rm = T)

DEF_1997 <-CF_VT_DEF[c(181:192),c(1,3)]
DEF_1997$mean <- mean(DEF_1997$DEF, na.rm = T)

DEF_1998 <-CF_VT_DEF[c(193:204),c(1,3)]
DEF_1998$mean <- mean(DEF_1998$DEF, na.rm = T)

DEF_1999 <-CF_VT_DEF[c(205:216),c(1,3)]
DEF_1999$mean <- mean(DEF_1999$DEF, na.rm = T)

DEF_2000 <-CF_VT_DEF[c(217:228),c(1,3)]
DEF_2000$mean <- mean(DEF_2000$DEF, na.rm = T)

DEF_2001 <-CF_VT_DEF[c(229:240),c(1,3)]
DEF_2001$mean <- mean(DEF_2001$DEF, na.rm = T)

DEF_2002 <-CF_VT_DEF[c(241:252),c(1,3)]
DEF_2002$mean <- mean(DEF_2002$DEF, na.rm = T)

DEF_2003 <-CF_VT_DEF[c(253:264),c(1,3)]
DEF_2003$mean <- mean(DEF_2003$DEF, na.rm = T)

DEF_2004 <-CF_VT_DEF[c(265:276),c(1,3)]
DEF_2004$mean <- mean(DEF_2004$DEF, na.rm = T)

DEF_2005 <-CF_VT_DEF[c(277:288),c(1,3)]
DEF_2005$mean <- mean(DEF_2005$DEF, na.rm = T)

DEF_2006 <-CF_VT_DEF[c(289:300),c(1,3)]
DEF_2006$mean <- mean(DEF_2006$DEF, na.rm = T)

DEF_2007 <-CF_VT_DEF[c(301:312),c(1,3)]
DEF_2007$mean <- mean(DEF_2007$DEF, na.rm = T)

DEF_2008 <-CF_VT_DEF[c(313:324),c(1,3)]
DEF_2008$mean <- mean(DEF_2008$DEF, na.rm = T)

DEF_2009 <-CF_VT_DEF[c(325:336),c(1,3)]
DEF_2009$mean <- mean(DEF_2009$DEF, na.rm = T)

DEF_2010 <-CF_VT_DEF[c(337:348),c(1,3)]
DEF_2010$mean <- mean(DEF_2010$DEF, na.rm = T)

DEF_2011 <-CF_VT_DEF[c(349:360),c(1,3)]
DEF_2011$mean <- mean(DEF_2011$DEF, na.rm = T)

DEF_2012 <-CF_VT_DEF[c(361:372),c(1,3)]
DEF_2012$mean <- mean(DEF_2012$DEF, na.rm = T)

DEF_2013 <-CF_VT_DEF[c(373:384),c(1,3)]
DEF_2013$mean <- mean(DEF_2013$DEF, na.rm = T)

DEF_2014 <-CF_VT_DEF[c(385:396),c(1,3)]
DEF_2014$mean <- mean(DEF_2014$DEF, na.rm = T)

DEF_2015 <-CF_VT_DEF[c(397:408),c(1,3)]
DEF_2015$mean <- mean(DEF_2015$DEF, na.rm = T)

DEF_2016 <-CF_VT_DEF[c(409:415),c(1,3)]
DEF_2016$mean <- mean(DEF_2016$DEF, na.rm = T)

#combine the data of DEF
CF_DEF <- bind_rows(DEF_1982,DEF_1983,DEF_1984,DEF_1985,DEF_1986,DEF_1987,
                    DEF_1988,DEF_1989,DEF_1990,DEF_1991,DEF_1992,DEF_1993,
                    DEF_1994,DEF_1995,DEF_1996,DEF_1997,DEF_1998,DEF_1999,
                    DEF_2000,DEF_2001,DEF_2002,DEF_2003,DEF_2004,DEF_2005,
                    DEF_2006,DEF_2007,DEF_2008,DEF_2009,DEF_2010,DEF_2011,
                    DEF_2012,DEF_2013,DEF_2014,DEF_2015,DEF_2016)

#CO2, H2O and CO2/H2O gas ratios
CF_Gas_BG$'CO2/H2O' <- (CF_Gas_BG$CO2/CF_Gas_BG$H2O)
CF_Gas_BN$'CO2/H2O' <- (CF_Gas_BN$CO2/CF_Gas_BN$H2O)
CF_Gas_Pisc$'CO2/H2O' <-(CF_Gas_Pisc$CO2/CF_Gas_Pisc$H2O)

#H2O/CO2
CF_Gas_BG$'H2O/CO2' <- (CF_Gas_BG$H2O/CF_Gas_BG$CO2)
CF_Gas_BN$'H2O/CO2' <- (CF_Gas_BN$H2O/CF_Gas_BN$CO2)
CF_Gas_Pisc$'H2O/CO2' <-(CF_Gas_Pisc$H2O/CF_Gas_Pisc$CO2)

#H2O/H2
CF_Gas_BG$'H2O/H2' <- (CF_Gas_BG$H2O/CF_Gas_BG$H2)
CF_Gas_BN$'H2O/H2' <- (CF_Gas_BN$H2O/CF_Gas_BN$H2)
CF_Gas_Pisc$'H2O/H2' <-(CF_Gas_Pisc$H2O/CF_Gas_Pisc$H2)

#CH4/H2O gas ratio
CF_Gas_BG$'CH4/H2O' <- (CF_Gas_BG$CH4/CF_Gas_BG$H2O)
CF_Gas_BN$'CH4/H2O' <- (CF_Gas_BN$CH4/CF_Gas_BN$H2O)
CF_Gas_Pisc$'CH4/H2O' <-(CF_Gas_Pisc$CH4/CF_Gas_Pisc$H2O)

#CO2/CH4 and CO2/H2S gas ratios
CF_Gas_BG$'CO2/CH4' <- (CF_Gas_BG$CO2/CF_Gas_BG$CH4)
CF_Gas_BN$'CO2/CH4' <- (CF_Gas_BN$CO2/CF_Gas_BN$CH4)
CF_Gas_Pisc$'CO2/CH4' <-(CF_Gas_Pisc$CO2/CF_Gas_Pisc$CH4)

CF_Gas_BG$'CO2/H2S' <- (CF_Gas_BG$CO2/CF_Gas_BG$H2S)
CF_Gas_BN$'CO2/H2S' <- (CF_Gas_BN$CO2/CF_Gas_BN$H2S)
CF_Gas_Pisc$'CO2/H2S' <-(CF_Gas_Pisc$CO2/CF_Gas_Pisc$H2S)

##Mean value of CO2
CF_Gas_BG$'CO2 mean'<-mean(CF_Gas_BG$'CO2')
CF_Gas_BN$'CO2 mean'<-mean(CF_Gas_BN$'CO2')
CF_Gas_Pisc$'CO2 mean'<-mean(CF_Gas_Pisc$'CO2')

##Mean value of N2
CF_Gas_BG$'N2 mean' <- mean(CF_Gas_BG$'N2',na.rm = T)
CF_Gas_BN$'N2 mean'<-mean(CF_Gas_BN$'N2',na.rm = T)
CF_Gas_Pisc$'N2 mean'<-mean(CF_Gas_Pisc$'N2',na.rm = T)

##Standard deviation of CO2
CF_Gas_BG$'CO2 sd'<-sd(CF_Gas_BG$'CO2')
CF_Gas_BN$'CO2 sd'<-sd(CF_Gas_BN$'CO2')
CF_Gas_Pisc$'CO2 sd'<-sd(CF_Gas_Pisc$'CO2')

##Standard deviation of N2
CF_Gas_BG$'N2 sd' <- sd(CF_Gas_BG$'N2',na.rm = T)
CF_Gas_BN$'N2 sd'<-sd(CF_Gas_BN$'N2',na.rm = T)
CF_Gas_Pisc$'N2 sd'<-sd(CF_Gas_Pisc$'N2',na.rm = T)

##The Z-score of CO2
CF_Gas_BG$'Z_score' <- 
  {(CF_Gas_BG$CO2)-(CF_Gas_BG$`CO2 mean`)}/(CF_Gas_BG$`CO2 sd`)
CF_Gas_BN$'Z_score' <-
  {(CF_Gas_BN$CO2)-(CF_Gas_BN$`CO2 mean`)}/(CF_Gas_BN$`CO2 sd`)
CF_Gas_Pisc$'Z_score' <-
  {(CF_Gas_Pisc$CO2)-(CF_Gas_Pisc$`CO2 mean`)}/(CF_Gas_Pisc$`CO2 sd`)

#Rename the Z-score to Z_score of CO2
CF_Gas_BG <- rename(CF_Gas_BG,'Z_score of CO2'='Z_score')
CF_Gas_BN <- rename(CF_Gas_BN,'Z_score of CO2'='Z_score')
CF_Gas_Pisc <- rename(CF_Gas_Pisc,'Z_score of CO2'='Z_score')

#The Z-score of N2
CF_Gas_BG$'Z_score of N2' <- 
  {(CF_Gas_BG$N2)-(CF_Gas_BG$`N2 mean`)}/(CF_Gas_BG$`N2 sd`)
CF_Gas_BN$'Z_score of N2' <-
  {(CF_Gas_BN$N2)-(CF_Gas_BN$`N2 mean`)}/(CF_Gas_BN$`N2 sd`)
CF_Gas_Pisc$'Z_score of N2' <-
  {(CF_Gas_Pisc$N2)-(CF_Gas_Pisc$`N2 mean`)}/(CF_Gas_Pisc$`N2 sd`)

#N2/CO2
CF_Gas_BG$'N2/CO2' <- (CF_Gas_BG$N2/CF_Gas_BG$CO2)
CF_Gas_BN$'N2/CO2' <- (CF_Gas_BN$N2/CF_Gas_BN$CO2)
CF_Gas_Pisc$'N2/CO2' <-(CF_Gas_Pisc$N2/CF_Gas_Pisc$CO2)

###############################################################################

#Graphs in Bocca Grande fumarole----
#Convert Character to Date in DATE
CF_Gas_BG$DATE <- dmy(CF_Gas_BG$DATE)
class(CF_Gas_BG$DATE)

CF_VT$DATE <- dmy(CF_VT$DATE)

CF_DEF$DATE <- ymd(CF_DEF$DATE)
class(CF_DEF$DATE)

CF_VT_DEF$DATE <- dmy(CF_VT_DEF$DATE)
class(CF_VT_DEF$DATE)

#The relationship of Temperature and Deformation at Bacca Grande
`TEMP` <-xyplot(CF_Gas_BG$`TEMP`~ CF_Gas_BG$DATE, CF_Gas_BG,
                   xlab=list("Date(year)"),
                   ylab = ("Celsius(°C)"),
                   main="The relationship between Temperature and Deformation at Bocca Grande ",
                   type="l",lwd=2,col="#FF0000")

`Deformation` <- xyplot(CF_DEF$DEF ~ CF_DEF$DATE, CF_DEF, 
                        xlab=list("Date"),
                        ylab = ("Elevation(cm)"),
                        type = "p", col="#7D26CD")

plot_t_d_b<-update(doubleYScale(`TEMP`,`Deformation` ,text = c("Temperature", "Deformation"), add.ylab2 = T), use.style=FALSE,par.settings = simpleTheme(col = c('#FF0000','#7D26CD'),lty =2:1,pch =15))

print(plot_t_d_b)

#The relationship of Temperature and VT at Bacca Grande
`TEMP` <-xyplot(CF_Gas_BG$`TEMP`~ CF_Gas_BG$DATE, CF_Gas_BG,
                xlab=list("Date(year)"),
                ylab = ("Celsius(°C)"),
                main="The relationship between Temperature and VT earthquakes at Bocca Grande ",
                type="l",lwd=2,col="#FF0000")

`VT` <- xyplot(CF_VT$VT ~ CF_VT$DATE, CF_VT, 
                        xlab=list("Date"),
                        ylab = ("VT earthquakes"), 
                        type = "l", lwd=2, col="#27408B")

plot_t_v_b<-update(doubleYScale(`TEMP`, `VT` ,text = c("Temperature", "VT earthquakes"), add.ylab2 = T), use.style=FALSE,par.settings = simpleTheme(col = c('#FF0000','#27408B'),lty = 2:1))

print(plot_t_v_b)

#stack (temp vs deformation) + (Temp vs VT) in BG
plotstack_bg <- grid.arrange(plot_t_d_b, plot_t_v_b,heights = c(1/2, 1/2))
print(plotstack_bg)

#CO2, H2O graph
plot1 <- ggplot(data = CF_Gas_BG, aes(x = DATE)) + geom_point(aes(y = H2O, colour = 'H2O'), na.rm = TRUE,pch=15) + geom_point(aes(y = CO2, colour = 'CO2'), na.rm = TRUE,pch=15) +
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)', colour = '') +
  scale_colour_manual(labels = c(expression(paste("CO"[2]),"H"[2]*"O")),values = c('red', 'blue')) + ggtitle('Carbon dioxide and water vapour concentration at Bocca Grande fumarole (1983-2016)')  +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot1)

#CO2/H2O gas ratios graph
plot2 <-  ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = CO2/H2O), na.rm =
              TRUE, colour = 'red',shape=15) +
   labs(x = 'Date(year)', y = expression('CO'[2]/'H'[2]*'O')) + ggtitle('The ratio of Carbon dioxide over water vapour concentration at Bocca Grande fumarole (1983-2016)')+
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot2)

#The comparison of CO2/H2O and Deformation in BG
`CO2/H2O_b` <-xyplot(CF_Gas_BG$`CO2/H2O`~ CF_Gas_BG$DATE, CF_Gas_BG,
                   xlab=list("Date(year)"),
                   ylab = expression("CO"[2]/"H"[2]*"O ratio"),
                   main=expression(bold(paste("The relationship between CO"[2]/"H"[2],"O and Deformation at Bocca Grande"))),
                   type="l",lwd=2,col='#FF0000')

`Deformation` <- xyplot(CF_DEF$DEF ~ CF_DEF$DATE, CF_DEF, 
                        xlab=list("Date"),
                        ylab = ("Elevation(cm)"),
                        type = "p", lwd=2, col='#27408B')

plot_co2_h2o_defo<-update(doubleYScale(`CO2/H2O_b`, `Deformation` ,text = c(expression("CO"[2]/"H"[2]*"O ratio"), "Deformation"), add.ylab2 = T), use.style=FALSE,par.settings = simpleTheme(col = c('#FF0000','#27408B'),lty = 2:1,pch=15))

print(plot_co2_h2o_defo)

##Create CO2 in BG  
plot3_0 <- ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = CO2), na.rm =
              TRUE, colour = '154',shape=15)+labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle('Carbon dioxide concentration at Bocca Grande fumarole (1983-2016)') +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot3_0)

#CH4 in BG
plot3_1<-ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = CH4), na.rm = TRUE, colour = '255',shape=15)+
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle('Methane concentration at Bocca Grande fumarole (1983-2016)')+
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot3_1)

#H2S in BG
plot3_2<-ggplot()+
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = H2S), na.rm = TRUE, colour = '148',shape= 15)+
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle('Hydrogen sulfide concentration at Bocca Grande fumarole (1983-2016)') +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot3_2)

#stack CO2,CH4 and H2S
plot3_0/plot3_1/plot3_2
plot3<-plot3_0/plot3_1/plot3_2
print(plot3)

#CO2/CH4 in BG
plot4_title=expression(bold(paste(CO[2]/CH[4]," ratios at Bocca Grande fumarole(1983-2016)")))
plot4 <-  ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = CO2/CH4), na.rm =
              TRUE, colour = '#FF6EB4',shape=15) +
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle(plot4_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5))

print(plot4)

#CO2/H2S in BG
plot5_title=expression(bold(paste(CO[2]/H[2],"S ratios at Bocca Grande fumarole(1983-2016)")))
plot5<- ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = `CO2/H2S`), na.rm =
              TRUE, colour = '#00BFFF', shape=15)+
  labs(x ='Date(year)', y = 'Gas species concentration (ppm)') + ggtitle(plot5_title)+
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5))

print(plot5)

#stack CO2/CH4+C)2/H2S
plot4/plot5
plot6<-plot4/plot5
print(plot6)

#CO2 and Deformation at BG
`CO2_b` <-xyplot(CF_Gas_BG$`CO2`~ CF_Gas_BG$DATE, CF_Gas_BG,
                   xlab=list("Date(year)"),
                   ylab = expression("                       CO"[2]),
                   main=expression(bold(paste("The relationship between CO"[2]~ "and Deformation at Bocca Grande"))),
                   type="l",lwd=3,col="#FF0000")

`Deformation` <- xyplot(CF_DEF$DEF ~ CF_DEF$DATE, CF_DEF, 
                        xlab=list("Date"),
                        ylab = ("Elevation(cm)"),
                        type = "p", lwd=2, col="#27408B")

plot_co2_defo<-update(doubleYScale(`CO2_b`, `Deformation` ,text = c(expression("CO"[2]), "Deformation"), add.ylab2 = T), use.style=FALSE,par.settings = simpleTheme(col = c('#FF0000','#27408B'),lty = 2:1,pch=15))

print(plot_co2_defo)

#Normalised N2 and CO2
plot7 <-   ggplot(data = CF_Gas_BG,aes(x = DATE)) +
  geom_point(aes(y=`Z_score of CO2`,colour='Z-score of CO2'),na.rm = TRUE,shape=15) +
  geom_point(aes(y=`Z_score of N2`,colour='Z-score of N2'),na.rm = TRUE,shape=15)+scale_colour_manual(values = c('red', 'blue'),labels=c(expression("Z scores of CO"[2],"Z   scores of N"[2])))+
  labs(x = 'Date(year)', y = 'Z scores',colour = '') + ggtitle('Z scores at Bocca Grande fumarole (1983-2016)') +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot7)

#combine plot and plot(the relationship with deformation)
plot7/plot8
plot9<-plot7/plot8
print(plot9)

###############################################################################

##Deformation ----
plot8 <- ggplot(data = CF_DEF, aes(x = DATE, y = DEF, group = 1))+
  geom_line(data = CF_DEF[!is.na(CF_DEF$DEF),], na.rm = TRUE, linetype = 'solid', size = 1, colour='blue') +
  geom_line(arrow = arrow(length=unit(0.30,"cm"), ends="first", type = "closed"), size = 3)+geom_segment(x=as.Date('1981-01-01'),y=0,xend=as.Date("1981-01-01"),yend=179,
  lineend = "round",linejoin = "round",size=0.5,linetype=2,
  arrow = arrow(length = unit(0.75,"cm")),colour = "#EE2C2C")+
  geom_curve(x=as.Date('1985-01-01'),y=180,xend=as.Date("2005-02-01"),yend=90,
               lineend = "round",size=0.5,linetype=2,curvature=-0.2,
               arrow = arrow(length = unit(0.75,"cm")),colour = "#008B8B")+
  geom_curve(x=as.Date('2005-10-01'),y=90,xend=as.Date("2016-07-01"),yend=130,
               lineend = "round",size=0.5,linetype=2,
               arrow = arrow(length = unit(0.75,"cm")),colour = "#CD3333")+
  geom_point(na.rm = TRUE, size = 0.5,colour='blue') +
  labs(x = 'Date (year)', y = 'Elevation (cm)') +
  ggtitle('Ground deformation at Campi Flegrei (1982-2016)') + scale_y_continuous(limits = c(0, 200), expand = c(0, 0)) + scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01', '2018-01-01'))) + theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot8)

ggplotly(plot8)

#VT
plot8_0 <- ggplot(data = CF_VT_DEF, aes(x = DATE, y = VT)) + geom_bar(stat = 'identity', na.rm = TRUE, colour = 'red',width = 0.5) +
  labs(x = 'Date (year)', y = 'Monthly number of VT earthquakes') +
  ggtitle('Monthly number of VT earthquakes at Campi Flegrei (1982-2016)') +
  scale_y_continuous(limits = c(0, 1250), expand = c(0, 0)) + scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01', '2018-01-01'))) + theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot8_0)

ggplotly(plot8_0)

#Compare the VT and deformation
`VT` <-xyplot(CF_VT_DEF$VT~ CF_VT_DEF$DATE, CF_VT_DEF,
                             xlab=list("Date(year)"),
                             ylab = ("VT"),
                             main="The relationship between VT and Deformation",
                             type="b",lwd=2,col="#FF0000")
                             

`Deformation` <- xyplot(CF_VT_DEF$`DEF` ~  CF_VT_DEF$`DATE`, CF_VT_DEF, 
                                       xlab=list("Date"),
                                       ylab = ("Elevation(cm)"),
                                       type = "b", lwd=2, col="#27408B")

plot_vt_defor<-update(doubleYScale(`VT`, `Deformation` ,text = c(expression("Volcano-tectonic seismicity"), "Deformation"), add.ylab2 = T), use.style=FALSE,par.settings = simpleTheme(col = c('#FF0000','#27408B'),lty = 2:1,pch=15))

print(plot_vt_defor)

#The relationship bewteen Deformation and VT at Campi Flegrei (1982-2016)
vt_defor_plot<- ggplot(data = CF_VT_DEF, aes(x = DEF, y = VT, group = 1))+
  geom_line(data = CF_VT_DEF[!is.na(CF_VT_DEF$DEF),], na.rm = TRUE, linetype = 'solid', size = 1, colour='blue')+
  geom_point(data = CF_VT_DEF[!is.na(CF_VT_DEF$DEF),], na.rm = TRUE, shape=18, size = 3, colour='red')+
  labs(x = 'Uplift (cm)', y = 'Monthly number of VT earthquakes') +
  ggtitle('The relationship bewteen Deformation and VT at Campi Flegrei (1982-2016)') +
  scale_y_continuous(limits = c(0, 1250), expand = c(0, 0)) + theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(vt_defor_plot)

#Mean VT and deformation rates
mean_VT_DEF_rates<- left_join(CF_VT,CF_DEF,by='DATE')
View(mean_VT_DEF_rates)

class(mean_VT_DEF_rates$DATE)

mean_VT_DEF_rates<-rename(mean_VT_DEF_rates,'The average of VT' = 'mean.x')
mean_VT_DEF_rates<-rename(mean_VT_DEF_rates,'The average of DEF' = 'mean.y')

#Relationship between average VT and average deformation
`The average of VT` <-xyplot(mean_VT_DEF_rates$`The average of VT` ~ mean_VT_DEF_rates$DATE, mean_VT_DEF_rates,
                   xlab=list("Date(year)"),
                   ylab = ("The average of VT"),
                   main="Relationship between average VT and average deformation",
                   type="b",lwd=2,col='#FF0000')

`The average of Deformation` <- xyplot(mean_VT_DEF_rates$`The average of DEF` ~ mean_VT_DEF_rates$DATE, mean_VT_DEF_rates, 
                        xlab=list("Date"),
                        ylab = ("The average of elevation(cm)"),
                        type = "b", lwd=2, col='#27408B')

plot_mean_vt_defor<-update(doubleYScale(`The average of VT`, `The average of Deformation` ,text = c(expression("VT"), "Deformation"), add.ylab2 = T), use.style=FALSE,par.settings = simpleTheme(col = c('#FF0000','#27408B'),lty = 2:1))

print(plot_mean_vt_defor)

###############################################################################

#Graphs in Bocca Nuova fumarole----

##Convert Character to Date
CF_Gas_BN$DATE <- dmy(CF_Gas_BN$DATE)
class(CF_Gas_BN$DATE)

##The relationship of Temperature and Deformation at Bacca Nuova
`TEMP_BN` <-xyplot(CF_Gas_BN$`TEMP`~ CF_Gas_BN$DATE, CF_Gas_BN,
                xlab=list("Date(year)"),
                ylab = ("Celsius(°C)"),
                main="The relationship between Temperature and Deformation at Bocca Nuova ",
                type="l",lwd=2,col="#FF0000")

`Deformation` <- xyplot(CF_DEF$DEF ~ CF_DEF$DATE, CF_DEF, 
                        xlab=list("Date"),
                        ylab = ("Elevation(cm)"),
                        type = "b", lwd=2, col="#27408B")

plot_t_d_bn<-update(doubleYScale(`TEMP_BN`, `Deformation` ,text = c("Temperature", "Deformation"), add.ylab2 = T), use.style=FALSE,par.settings = simpleTheme(col = c('#FF0000','#27408B'),lty = 2:1,pch=15))

print(plot_t_d_bn)

##The relationship of Temperature and VT at Bacca Nuova
`TEMP_BN` <-xyplot(CF_Gas_BN$`TEMP`~ CF_Gas_BN$DATE, CF_Gas_BN,
                xlab=list("Date(year)"),
                ylab = ("Celsius(°C)"),
                main="The relationship between Temperature and VT earthquakes at Bocca Nuova ",
                type="l",lwd=2,col="#FF0000")

`VT` <- xyplot(CF_VT$VT ~ CF_VT$DATE, CF_VT, 
               xlab=list("Date"),
               ylab = ("VT earthquakes"), 
               type = "l", lwd=2, col="#27408B")

plot_t_v_bn<-update(doubleYScale(`TEMP_BN`, `VT` ,text = c("Temperature", "VT earthquakes"), add.ylab2 = T), use.style=FALSE,par.settings = simpleTheme(col = c('#FF0000','#27408B'),lty = 2:1))

print(plot_t_v_bn)

#stack (Temp vs Deformation) and (Temp vs VT) in BN
plotstack_bn<- grid.arrange(plot_t_d_bn, plot_t_v_bn,heights = c(1/2, 1/2))

##Create a CO2, H2O graph
plot12 <- ggplot(data = CF_Gas_BN, aes(x= DATE)) +
  geom_point(aes(y = H2O, colour = 'H2O'), na.rm =
              TRUE,shape=15) +
  geom_point(aes(y = CO2, colour = 'CO2'), na.rm = TRUE,shape=15) +
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)', colour = '') + ggtitle('Carbon dioxide and water vapour concentration at Bocca Nuova fumarole (1995-2016)')+
  scale_colour_manual(values = c('red','blue'),labels=c(expression("CO"[2],"H"[2]*"O")))+
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1995-01-01',  '2016-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot12)

##Create a CO2/H2O gas ratios graph in BN
plot13 <-  ggplot() +
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = CO2/H2O), na.rm =
              TRUE, colour = 'red',shape=15) +
  labs(x = 'Date(year)', y =expression('CO'[2]~"/H"[2]*'O')) + ggtitle('The ratio of Carbon dioxide over water vapour concentration at Bocca Nuova fumarole (1995-2016)') +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1995-01-01',  '2016-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot13)

#The comparison of CO2/H2O and Deformation in BN
`CO2/H2O_BN` <-xyplot(CF_Gas_BN$`CO2/H2O`~ CF_Gas_BN$DATE, CF_Gas_BN,
                   xlab=list("Date(year)"),
                   ylab = expression("CO"[2]~"/H"[2]*"O ratio"),
                   main=expression(bold(paste("The relationship between CO"[2]*"/H"[2]*"O and Deformation"))),
                   type="l",lwd=2,col="#FF0000")

`Deformation` <- xyplot(CF_DEF$DEF ~ CF_DEF$DATE, CF_DEF, 
                        xlab=list("Date(year)"),
                        ylab = ("Elevation(cm)"),
                        type = "p", lwd=2, col="#27408B")

plot_co2_defo_bn<-update(doubleYScale(`CO2/H2O_BN`, `Deformation` ,
                                      text = c(expression("CO"[2]*"/H"[2]*"O ratio", "VT earthquakes")), add.ylab2 = T), use.style=FALSE,par.settings = simpleTheme(col = c('#FF0000','#27408B'),lty = 2:1,pch=15))

print(plot_co2_defo_bn)

#Create CO2 in BN
plot14_0 <- ggplot() +
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = CO2), na.rm =
              TRUE, colour = 'red',shape=15)+labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle('Carbon dioxide concentration at Bocca Nuova fumarole (1995-2016)') +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1995-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot14_0)

#CH4 in BN
plot14_1<-ggplot() +
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = CH4), na.rm = TRUE, colour = 'blue',shape=15)+
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle('Methane concentration at Bocca Nuova fumarole (1995-2016)')+
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1995-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot14_1)

#H2S in BN
plot14_2<-ggplot()+
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = H2S), na.rm = TRUE, colour = 'green',shape=15)+
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle('Hydrogen sulfide concentration at Bocca Nuova fumarole (1995-2016)')+
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1995-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot14_2)

#stack CO2+CH4+H2S
plot14_0/plot14_1/plot14_2
plot14<-plot14_0/plot14_1/plot14_2
print(plot14)

#CO2/CH4gas ratios by time series
plot15_title=expression(bold(paste(CO[2]/CH[4]," ratios at Bocca Nuova fumarole (1995-2016)")))
plot15 <-  ggplot() +
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = CO2/CH4), na.rm =
              TRUE, colour = 'red',shape=15) +
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle(plot15_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1995-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot15)

#CO2/H2S gas ratios
plot16_title=expression(bold(paste("CO"[2]/"H"[2]*"S ratios at Bocca Nuova fumarole (1995-2016)")))
plot16<- ggplot() +
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = `CO2/H2S`), na.rm =
              TRUE, colour = 'blue',shape=15)+
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle(plot16_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1995-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot16)

#stack (co2/ch4)+(CO2/H2S)
plot15/plot16
plot17<-plot15/plot16
print(plot17)

##Normalised N2 and CO2 in BN
plot18 <-  ggplot(data = CF_Gas_BN,aes(x = DATE)) +
geom_point(aes(y=`Z_score of CO2`,colour='Z-score of CO2'),na.rm = TRUE,shape=15) +
geom_point(aes(y=`Z_score of N2`,colour='Z-score of N2'),na.rm = TRUE,shape=15)+
  labs(x = 'Date(year)', y = 'Z scores', colour = '') + ggtitle('Z scores at Bocca Nuova fumarole (1995-2016)') +scale_colour_manual(values = c('red','blue'),labels=c(expression("Z scores of CO"[2],"Z    scores of N"[2])))+
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1995-01-01',  '2018-01-01')))+theme_test()+
  theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot18)

#deformation
plot18_1 <- ggplot(data = CF_DEF, aes(x = DATE, y = DEF, group = 1))+
  geom_line(data = CF_DEF[!is.na(CF_DEF$DEF),], na.rm = TRUE, linetype = 'solid', size = 1, colour='blue') +
  geom_point(na.rm = TRUE, size = 0.5,colour='blue') +
  geom_segment(x=as.Date('1995-01-01'),y=120,xend=as.Date("2005-02-01"),yend=90,
               lineend = "round",linejoin = "round",size=0.5,linetype=2,
               arrow = arrow(length = unit(0.75,"cm")),colour = "#008B8B")+
  geom_segment(x=as.Date('2005-10-01'),y=90,xend=as.Date("2016-07-01"),yend=130,
               lineend = "round",linejoin = "round",size=0.5,linetype=2,
               arrow = arrow(length = unit(0.75,"cm")),colour = "#CD3333")+
  labs(x = 'Date(year)', y = 'Elevation (cm)') +
  ggtitle('Ground deformation at Campi Flegrei (1995-2016)') + scale_y_continuous(limits = c(0, 200), expand = c(0, 0)) + scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1995-01-01', '2018-01-01'))) + theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot18_1)

##combine plot and plot(the relationship with deformation)
plot18/plot18_1
plot18_0 <-plot18/plot18_1
print(plot18_0)

###############################################################################

#Graphs in Pisciarelli fumarole----

##Convert Character to Date
CF_Gas_Pisc$DATE <- dmy(CF_Gas_Pisc$DATE)
class(CF_Gas_Pisc$DATE)

##The relationship of Temperature and Deformation at Pisciarelli fumarole
`TEMP_Pisc` <-xyplot(CF_Gas_Pisc$`TEMP`~ CF_Gas_Pisc$DATE, CF_Gas_Pisc,
                   xlab=list("Date(year)"),
                   ylab = ("Celsius(°C)"),
                   main="The relationship between Temperature and Deformation at Pisciarelli",
                   type="l",lwd=2,col="#FF0000")

`Deformation` <- xyplot(CF_DEF$DEF ~ CF_DEF$DATE, CF_DEF, 
                        xlab=list("Date"),
                        ylab = ("Elevation(cm)"),
                        type = "p", lwd=2, col="#27408B")

plot_t_d_p<-update(doubleYScale(`TEMP_Pisc`, `Deformation` ,text = c("Temperature", "Deformation"), add.ylab2 = T), use.style=FALSE,par.settings = simpleTheme(col = c('#FF0000','#27408B'),lty = 2:1,pch=15))

print(plot_t_d_p)

##The relationship of Temperature and VT at Pisciarelli fumarole
`TEMP_Pisc` <-xyplot(CF_Gas_Pisc$`TEMP`~ CF_Gas_Pisc$DATE, CF_Gas_Pisc,
                   xlab=list("Date(year)"),
                   ylab = ("Celsius(°C)"),
                   main="The relationship between Temperature and VT earthquakes at Pisciarelli fumarole",
                   type="l",lwd=2,col="#FF0000")

`VT` <- xyplot(CF_VT$VT ~ CF_VT$DATE, CF_VT, 
               xlab=list("Date"),
               ylab = ("VT earthquakes"),
               type = "l", lwd=2, col="#27408B")

plot_t_v_p<-update(doubleYScale(`TEMP_Pisc`, `VT` ,text = c("Temperature", "VT earthquakes"), add.ylab2 = T), use.style=FALSE,par.settings = simpleTheme(col = c('#FF0000','#27408B'),lty = 2:1,pch=15))

print(plot_t_v_p)

#stack (Temp vs defromation) + (Temp vs VT) in Pisc
plotstack_p <- grid.arrange(plot_t_d_p, plot_t_v_p,heights = c(1/2, 1/2))
print(plotstack_p)

##Create a CO2, H2O graph
plot19 <- ggplot(data = CF_Gas_Pisc ,aes(x = DATE) ) +
  geom_point(aes(y = H2O, colour = 'H2O'), na.rm =
              TRUE,shape=15) +
  geom_point(aes(y = CO2, colour = 'CO2'), na.rm = TRUE,shape=15) +
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)', colour='') +scale_colour_manual(values=c('red','blue'),labels=c(expression("CO"[2],"H"[2]*'O')))+ ggtitle('Carbon dioxide and water vapour concentration at Pisciarelli fumarole (1999-2011)') + scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1999-01-01', '2011-01-01'))) + theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot19)

ggplotly(plot19)

##Create a CO2/H2O gas ratios graph
plot20 <-  ggplot() +
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = CO2/H2O), na.rm =
              TRUE, colour = 'red',shape=15) +
  labs(x = 'Date(year)', y = expression('CO'[2]/'H'[2]*'O')) + ggtitle('The ratio of Carbon dioxide over water vapour concentration at Pisciarelli fumarole (199-2011)') + scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1999-01-01', '2011-01-01'))) + theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot20)

#The relationship between CO2/H2O and Deformation
`CO2/H2O_p` <-xyplot(CF_Gas_Pisc$`CO2/H2O`~ CF_Gas_Pisc$DATE, CF_Gas_Pisc,
                      xlab=list("Date(year)"),
                      ylab = expression("CO"[2]*"/H"[2]*"O ratio"),
                     main=expression(bold(paste("The relationship between CO"[2]*"/H"[2]*"O and Deformation"))),
                      type="l",lwd=2,col="#FF0000")

`Deformation` <- xyplot(CF_DEF$DEF ~ CF_DEF$DATE, CF_DEF, 
                        xlab=list("Date(year)"),
                        ylab = ("Elevation(cm)"),
                        type = "p", lwd=2, col="#27408B")

plot_co2_defo_p<-update(doubleYScale(`CO2/H2O_p`, `Deformation` ,
                                      text = c(expression("CO"[2]*"/H"[2]*"O ratio", "Deformation")), add.ylab2 = T), use.style=FALSE,par.settings = simpleTheme(col = c('#FF0000','#27408B'),lty = 2:1,pch=15))

print(plot_co2_defo_p)

#CO2 in pisc
plot21_0 <- ggplot() +
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = CO2), na.rm =
              TRUE, colour = 'red',shape=15)+labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle('Carbon dioxide concentration at Pisciarelli fumarole (1999-2011)') + scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1999-01-01', '2011-01-01'))) + theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot21_0)

#CH4 in pisc
plot21_1<-ggplot() +
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = CH4), na.rm =
              TRUE, colour = 'blue',shape=15)+
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle('Methane concentration at Pisciarelli fumarole (1999-2011)') + scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1999-01-01', '2011-01-01'))) + theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot21_1)

#H2S in pisc
plot21_2<-ggplot()+
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = H2S), na.rm = 
              TRUE, colour = 'green',shape=15)+
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle('Hydrogen sulfide concentration at Pisciarelli fumarole (1999-2011)') + scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1999-01-01', '2011-01-01'))) + theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot21_2)

#Stack co2+ch4+h2s
plot21_0/plot21_1/plot21_2
plot21<-plot21_0/plot21_1/plot21_2
print(plot21)

# CO2/CH4 gas ratios by time series
plot22_title=expression(bold(paste(CO[2]/CH[4]," ratios at Pisciarelli fumarole (1999-2011)")))
plot22 <-  ggplot() +
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = CO2/CH4), na.rm =
              TRUE, colour = 'red',shape=18) +
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle(plot22_title) + scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1999-01-01', '2011-01-01'))) + theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot22)

#CO2/H2S gas ratios
plot23_title=expression(bold(paste(CO[2]/H[2],"S"," ratios at Pisciarelli fumarole (1999-2011)")))
plot23<- ggplot() +
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = `CO2/H2S`), na.rm =
              TRUE, colour = 'blue',shape=18)+
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle(plot23_title)+ scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1999-01-01', '2011-01-01'))) + theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot23)

#stack CO2/CH4 CO2/H2S gas ratios
plot22/plot23
plot24<-plot22/plot23
print(plot24)

#Normalised N2 and CO2 in PISC
plot25 <-  ggplot(data = CF_Gas_Pisc,aes(x = DATE)) +
  geom_point(aes(y=`Z_score of CO2`,colour='Z-score of CO2'),na.rm = TRUE,shape=15) +
  geom_point(aes(y=`Z_score of N2`,colour='Z-score of N2'),na.rm = TRUE,shape=15)+
  labs(x = 'Date(year)', y = 'Z scores', colour = '') + ggtitle('Z scores at Pisciarelli fumarole (1999-2011)') +scale_colour_manual(values = c('red','blue'),labels=c(expression("Z scores of CO"[2],"Z    scores of N"[2])))+
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1999-01-01',  '2011-01-01')))+theme_test()+
  theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot25)

#deformation from 1999 to 2011
plot25_1 <- ggplot(data = CF_DEF, aes(x = DATE, y = DEF, group = 1))+
  geom_line(data = CF_DEF[!is.na(CF_DEF$DEF),], na.rm = TRUE, linetype = 'solid', size = 1, colour='blue') +
  geom_point(na.rm = TRUE, size = 0.5,colour='blue') +
  geom_segment(x=as.Date('2005-10-01'),y=90,xend=as.Date("2010-07-01"),yend=110,
               lineend = "round",linejoin = "round",size=0.5,linetype=2,
               arrow = arrow(length = unit(0.75,"cm")),colour = "#CD3333")+
  labs(x = 'Date(year)', y = 'Elevation (cm)') +
  ggtitle('Ground deformation at Campi Flegrei (1999-2011)') + scale_y_continuous(limits = c(0, 200), expand = c(0, 0)) + scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1999-01-01', '2011-01-01'))) + theme_test()+theme(plot.title = element_text(hjust = 0.5,face = "bold"))

print(plot25_1)

#combine plot and plot(the relationship with deformation)
plot25/plot25_1
plot25_0 <-plot25/plot25_1
print(plot25_0)

###############################################################################

#map----
register_google(key = 'AIzaSyD1nobEpoqk1AhtSt0c0sX5ofccmeeWIpk')

##The map of Campi Flegrei
`Campi Flegrei` <- get_googlemap('Campi Flegrei', 
                                 zoom = 16,
                                 maptype = 'hybrid'
                                 ) %>% 
  ggmap(extent = 'device')

print(`Campi Flegrei`)

###############################################################################

#combine three areas----

#combine co2/ch4 in three areas
plotco2ch4sum_title=expression(bold(paste(CO[2]/CH[4]," ratios at Campi Flegrei Caldera (1983-2016)")))
plotco2ch4sum<-  ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = CO2/CH4,col=FUMAROLE),na.rm =
               TRUE,shape=15) +
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = CO2/CH4,col=FUMAROLE), na.rm =
               TRUE,shape=15) +
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = CO2/CH4,col=FUMAROLE), na.rm =
              TRUE ,shape=15)+scale_color_manual(
                values = c('#104E8B','#FF3030','#228B22'),
                labels = c("BG","BN","Pisc"))+
  labs(x = 'Date (year)', y = expression(CO[2]/CH[4])) + ggtitle(plotco2ch4sum_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+
  theme(plot.title = element_text(hjust = 0.5),panel.background = element_rect(fill = 'white', colour ='black'),
 legend.key = element_rect(fill = 'white'))

print(plotco2ch4sum)

#compare the deformation and co2/ch4

plotco2ch4sum/plot8
com_de_co2_ch4<-plotco2ch4sum/plot8
print(com_de_co2_ch4)

#combine co2/h2o in three areas
plotco2h2osum_title=expression(bold(paste(CO[2]/H[2]*O," ratios at Campi Flegrei Caldera (1983-2016)")))
plotco2h2osum <-  ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = CO2/H2O,col=FUMAROLE),na.rm =
               TRUE,shape=15) +
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = CO2/H2O,col=FUMAROLE), na.rm =
               TRUE,shape=15) +
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = CO2/H2O,col=FUMAROLE), na.rm =
               TRUE ,shape=15)+
  scale_color_manual(
                 values = c('#104E8B','#FF3030','#228B22'),
                 labels = c("BG","BN","Pisc"))+
  labs(x = 'Date (year)', y = expression(CO[2]/H[2]*O)) + 
  ggtitle(plotco2h2osum_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+
  theme_test()+
  theme(plot.title = element_text(hjust = 0.5),panel.background = element_rect(fill = 'white', colour ='black'),legend.key = element_rect(fill = 'white'))

print(plotco2h2osum)

#compare the deformation and co2/h2o
plotco2h2osum/plot8
com_de_co2_h2o<-plotco2h2osum/plot8
print(com_de_co2_h2o)

#combine h2o/co2 in three areas
plot_h2o_co2_title=expression(bold(paste(H[2]*O/CO[2]~"ratios at Campi Flegrei Caldera (1983-2016)")))
plot_h2o_co2<-  ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = H2O/CO2,col=FUMAROLE),na.rm =
               TRUE,shape=15) +
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = H2O/CO2,col=FUMAROLE), na.rm =
               TRUE,shape=15) +
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = H2O/CO2,col=FUMAROLE), na.rm =
               TRUE ,shape=15)+
  scale_color_manual(
    values = c('#104E8B','#FF3030','#228B22'),
    labels = c("BG","BN","Pisc"))+
  labs(x = 'Date (year)', y = expression(H[2]*O/CO[2])) + ggtitle(plot_h2o_co2_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+
  theme_test()+
  theme(plot.title = element_text(hjust = 0.5),panel.background = element_rect(fill = 'white', colour ='black'),legend.key = element_rect(fill = 'white'))

print(plot_h2o_co2)

#compare the deformation and h2o/co2
plot_h2o_co2/plot8
com_de_h2o_co2<-plot_h2o_co2/plot8
print(com_de_h2o_co2)

#combine co2/h2s in three areas
plotco2_h2s_title=expression(bold(paste(CO[2]/H[2],"S ratios at Bocca Grande fumarole (1983-2016)")))
plotco2_h2s<- ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = `CO2/H2S`,col=FUMAROLE), na.rm =
               TRUE, shape=15)+
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = `CO2/H2S`,col=FUMAROLE), na.rm =
               TRUE, shape=15)+
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = `CO2/H2S`,col=FUMAROLE), na.rm =
               TRUE, shape=15)+
  scale_color_manual(
    values = c('#104E8B','#FF3030','#228B22'),
    labels = c("BG","BN","Pisc"))+
  labs(x = 'Date (year)', y = expression(CO[2]/H[2]*S)) + ggtitle(plotco2_h2s_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+
  theme_test()+
  theme(plot.title = element_text(hjust = 0.5),panel.background = element_rect(fill = 'white', colour ='black'),legend.key = element_rect(fill = 'white'))

print(plotco2_h2s)

#compare the deformation and co2/h2s 
plotco2_h2s/plot8
com_de_co2_h2s<-plotco2_h2s/plot8
print(com_de_co2_h2s)

#combine co2 in three areas
plotco2sum_title=expression(bold(paste(CO[2]~"concentration at Campi Flegrei Caldera (1983-2016)")))
plotco2sum <- ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = CO2,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = CO2,col=FUMAROLE), na.rm =
              TRUE,shape=15)+
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = CO2,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  scale_color_manual(
    values = c('#104E8B','#FF3030','#228B22'),
    labels = c("BG","BN","Pisc"))+
  labs(x = 'Date (year)', y = 'Gas species concentration (ppm)') + 
  ggtitle(plotco2sum_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+
  theme_test()+
  theme(plot.title = element_text(hjust = 0.5),legend.position = c(0.1,0.8),panel.background = element_rect(fill = 'white', colour ='black'),legend.key = element_rect(fill = 'white'))

print(plotco2sum)

#combine ch4 in three areas
plotch4sum_title=expression(bold(paste(CH[4]~"concentration at Campi Flegrei Caldera (1983-2016)")))
plotch4sum <- ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = CH4,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = CH4,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = CH4,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  scale_color_manual(
    values = c('#104E8B','#FF3030','#228B22'),
    labels = c("BG","BN","Pisc"))+
  labs(x = 'Date (year)', y = 'Gas species concentration (ppm)') + ggtitle(plotch4sum_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+
  theme_test()+
  theme(plot.title = element_text(hjust = 0.5),legend.position = c(0.9,0.86),panel.background = element_rect(fill = 'white', colour ='black'),legend.key = element_rect(fill = 'white'))

print(plotch4sum)

#combine H2S in three areas
ploth2ssum_title=expression(bold(paste(H[2]*"S concentration at Campi Flegrei Caldera (1983-2016)")))
ploth2ssum <- ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = H2S,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = H2S,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = H2S,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  scale_color_manual(
    values = c('#104E8B','#FF3030','#228B22'),
    labels = c("BG","BN","Pisc"))+
  labs(x = 'Date (year)', y = 'Gas species concentration (ppm)') + 
  ggtitle(ploth2ssum_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+
  theme_test()+
  theme(plot.title = element_text(hjust = 0.5),legend.position = c(0.9,0.86),panel.background = element_rect(fill = 'white', colour ='black'),
legend.key = element_rect(fill = 'white'))

print(ploth2ssum)

#combine H2O in three areas
ploth2osum_title=expression(bold(paste(H[2]*"O concentration at Campi Flegrei Caldera (1983-2016)")))
ploth2osum <- ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = H2O,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = H2O,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = H2O,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  scale_color_manual(
    values = c('#104E8B','#FF3030','#228B22'),
    labels = c("BG","BN","Pisc"))+
  labs(x = 'Date(year)', y = 'Gas species concentration (ppm)') + ggtitle(ploth2osum_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+
  theme(plot.title = element_text(hjust = 0.5),legend.position = c(0.9,0.86),panel.background = element_rect(fill = 'white', colour ='black'),legend.key = element_rect(fill = 'white'))

print(ploth2osum)

#H2O/H2 in three areas
h2o_h2_sum_title=expression(bold(paste(H[2]*O/H[2]~"concentration at Campi Flegrei Caldera (1983-2016)")))
h2o_h2_sum <- ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = H2O/H2,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = H2O/H2,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = H2O/H2,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  scale_color_manual(
    values = c('#104E8B','#FF3030','#228B22'),
    labels = c("BG","BN","Pisc"))+
  labs(x = 'Date (year)', y = expression(H[2]*O/H[2])) + 
  ggtitle(h2o_h2_sum_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+
  theme_test()+
  theme(plot.title = element_text(hjust = 0.5),legend.position = c(0.9,0.86),panel.background = element_rect(fill = 'white', colour ='black'),legend.key = element_rect(fill = 'white'))

print(h2o_h2_sum )

#compare the deformation and H2O/H2
h2o_h2_sum /plot8
com_de_h2o_h2_sum <-h2o_h2_sum /plot8
print(com_de_h2o_h2_sum)

#N2/CO2 in three areas
n2_co2_sum_title=expression(bold(paste(N[2]/CO[2]~"ratio at Campi Flegrei Caldera (1983-2016)")))
n2_co2_sum <- ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = N2/CO2,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = N2/CO2,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = N2/CO2,col=FUMAROLE), na.rm =
               TRUE,shape=15)+
  scale_color_manual(
    values = c('#104E8B','#FF3030','#228B22'),
    labels = c("BG","BN","Pisc"))+
  labs(x = 'Date (year)', y = expression(N[2]/CO[2])) + ggtitle(n2_co2_sum_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+
theme(plot.title = element_text(hjust = 0.5),legend.position = c(0.9,0.86),
      panel.background = element_rect(fill = 'white', colour ='black'),
legend.key = element_rect(fill = 'white'))

print(n2_co2_sum )

#compare the deformation and N2/CO2
n2_co2_sum  /plot8
com_de_n2_co2_sum  <-n2_co2_sum  /plot8
print(com_de_n2_co2_sum )

# stack three areas(co2/ch4)
plotco2ch4sum_title=expression(bold(paste(CO[2]/CH[4]," ratios at Campi Flegrei Caldera (1983-2016)")))
plotco2ch4sum_stack<-  ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = CO2/CH4,col=FUMAROLE),na.rm =
               TRUE,shape=15) +
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = CO2/CH4,col=FUMAROLE), na.rm =
               TRUE,shape=15) +
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = CO2/CH4,col=FUMAROLE), na.rm =
               TRUE ,shape=15)+scale_color_manual(
                 values = c('#104E8B','#FF3030','#228B22'),
                 labels = c("BG","BN","Pisc"))+
  labs(x = 'Date (year)', y = expression(CO[2]/CH[4])) + ggtitle(plotco2ch4sum_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+
  theme(plot.title = element_text(hjust = 0.5),panel.background = element_rect(fill = 'white', colour ='black'),
        legend.key = element_rect(fill = 'white'),legend.position = c(0.08, 0.7))

print(plotco2ch4sum_stack)

# stack three areas(co2/h2)
plotco2_h2s_title=expression(bold(paste(CO[2]/H[2],"S ratios at Bocca Grande fumarole (1983-2016)")))
plotco2_h2s_stack<- ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = `CO2/H2S`,col=FUMAROLE), na.rm =
               TRUE, shape=15)+
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = `CO2/H2S`,col=FUMAROLE), na.rm =
               TRUE, shape=15)+
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = `CO2/H2S`,col=FUMAROLE), na.rm =
               TRUE, shape=15)+
  scale_color_manual(
    values = c('#104E8B','#FF3030','#228B22'),
    labels = c("BG","BN","Pisc"))+
  labs(x = 'Date (year)', y = expression(CO[2]/H[2]*S)) + ggtitle(plotco2_h2s_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5),
 panel.background = element_rect(fill = 'white', colour ='black'),
legend.key = element_rect(fill = 'white'),legend.position = c(0.08, 0.7))

print(plotco2_h2s_stack)

# stack three areas(co2/h2o)
plotco2h2osum_title=expression(bold(paste(CO[2]/H[2]*O," ratios at Campi Flegrei Caldera (1983-2016)")))
plotco2h2osum_stack <-  ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = CO2/H2O,col=FUMAROLE),na.rm =
               TRUE,shape=15) +
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = CO2/H2O,col=FUMAROLE), na.rm =
               TRUE,shape=15) +
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = CO2/H2O,col=FUMAROLE), na.rm =
               TRUE ,shape=15)+
  scale_color_manual(
    values = c('#104E8B','#FF3030','#228B22'),
    labels = c("BG","BN","Pisc"))+
  labs(x = 'Date (year)', y = expression(CO[2]/H[2]*O)) + 
  ggtitle(plotco2h2osum_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5),
panel.background = element_rect(fill = 'white', colour ='black'),
legend.key = element_rect(fill = 'white'),legend.position = c(0.08, 0.7) )

print(plotco2h2osum_stack)

# stack three plots(co2/ch4+co2/h2s+co2/h2o)
print(plotco2ch4sum_stack)/print(plotco2_h2s_stack)/print(plotco2h2osum_stack)

#Z scores of three areas
print(plot7)/print(plot18)/print(plot25)

#temperature in three areas
temper_title=expression(bold("Tempeature at Campi Flegrei Caldera (1983-2016)"))
temperature_stack <-  ggplot() +
  geom_point(data = CF_Gas_BG, aes(x = DATE, y = TEMP,col=FUMAROLE),na.rm =
               TRUE,shape=15) +
  geom_line(data = CF_Gas_BG, aes(x = DATE, y = TEMP,col=FUMAROLE),na.rm =
               TRUE)+
  geom_curve(data = CF_Gas_BG,x=as.Date('1996-01-01'),y=165,xend=as.Date("2005-02-01"),yend=155,
                                lineend = "round",size=0.5,linetype=2,curvature=-0.2,
                                arrow = arrow(length = unit(0.75,"cm")),colour = "#104E8B")+
  geom_curve(data = CF_Gas_BG,x=as.Date('2006-02-01'),y=154,xend=as.Date("2017-02-01"),yend=165,
             lineend = "round",size=0.5,linetype=2,curvature=-0.2,
             arrow = arrow(length = unit(0.75,"cm")),colour = "#104E8B")+
  geom_point(data = CF_Gas_BN, aes(x = DATE, y = TEMP,col=FUMAROLE), na.rm =
               TRUE,shape=15) +
    geom_line(data = CF_Gas_BN, aes(x = DATE, y = TEMP,col=FUMAROLE),na.rm =
                                   TRUE)+
  geom_curve(data = CF_Gas_BN,x=as.Date('1999-01-01'),y=153,xend=as.Date("2005-02-01"),yend=136,
             lineend = "round",size=0.5,linetype=2,curvature=-0.2,
             arrow = arrow(length = unit(0.75,"cm")),colour = "#FF3030")+
  geom_curve(data = CF_Gas_BN,x=as.Date('2006-01-01'),y=135,xend=as.Date("2017-02-01"),yend=146,
             lineend = "round",size=0.5,linetype=2,curvature=-0.2,
             arrow = arrow(length = unit(0.75,"cm")),colour = "#FF3030")+
  
  geom_point(data = CF_Gas_Pisc, aes(x = DATE, y = TEMP,col=FUMAROLE), na.rm =
               TRUE ,shape=15)+
    geom_line(data = CF_Gas_Pisc, aes(x = DATE, y = TEMP,col=FUMAROLE),na.rm =
                                   TRUE)+
  geom_curve(data = CF_Gas_Pisc,x=as.Date('2002-12-01'),y=107,xend=as.Date("2005-02-01"),yend=90,
            lineend = "round",size=0.5,linetype=2,curvature=-0.2,
            arrow = arrow(length = unit(0.75,"cm")),colour = "#228B22")+
  geom_curve(data = CF_Gas_Pisc,x=as.Date('2006-02-01'),y=90,xend=as.Date("2011-02-01"),yend=108,
             lineend = "round",size=0.5,linetype=2,curvature=-0.2,
             arrow = arrow(length = unit(0.75,"cm")),colour = "#228B22")+
  scale_color_manual(
    values = c('#104E8B','#FF3030','#228B22'),
    labels = c("BG","BN","Pisc"))+
  labs(x = 'Date (year)', y = "Celsius(°C)") + ggtitle(temper_title) +
  scale_x_date(date_breaks = '3 years', date_labels = '%Y', limits = as.Date(c('1980-01-01',  '2018-01-01')))+theme_test()+theme(plot.title = element_text(hjust = 0.5),
panel.background = element_rect(fill = 'white', colour ='black'),
legend.key = element_rect(fill = 'white'),legend.position = c(0.08, 0.7) )

print(temperature_stack)