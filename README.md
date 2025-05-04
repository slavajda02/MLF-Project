# Machine learning fundamentals project
A GitHub repository containing the solution to a MPC-MLF university semester project.

## Introduction
We are all using our cell phones on a daily basis. These are our connections with family and friends, help us to search
for information, etc. The cell phone needs always be connected to the network, through a so-called base station (as
telecommunication engineers, we call them eNodeB’s or gNodeB’s).
But, there can be attackers taking advantage of security vulnerabilities in mobile networks. With the use of
specialized hardware and software, they can steal some informations, send malicious messages, or track users. One of
the known tools is a so-called False Base Station (FBS), sometimes called Rogue Base Station (RBS), or International
Mobile Subscriber Identifier (IMSI) Catcher. This is a device pretending as the legitimate base station and trying
the mobile terminals to connect.
The false base station thus needs to behave as the legitimate one, i.e., it needs to broadcast the same information
to the radio channel

## Assignment
The goal of your project is to classify whether there is only a legitimate base station (gNodeB) of T-mobile
operator (class 0) transmitting from a neighboring building or whether an attacker brought his transmitter into the
building and is trying to steal the information from users (class 1 and class 2). The false/attacker’s base station
could be located at one of two positions - class 1 corresponds to the first one, and class 2 corresponds to the second
one.