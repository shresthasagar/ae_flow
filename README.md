# Flow Autoencoder
Final Project of Deep Learning CS535, Spring 2021. "final.pdf" contains the written report of this project.

Flow Autoencoder is a deep generative model that maps the aggregated posterior of an autoencoder to a simple prior. It combines autoencoder and normalizing flow in order to map the latent distribution into a simple (e.g. standard normal) distribution. As a result, the flow and the decoder, together, define a generative model that can generate samples from the learnt data distribution with a forward pass. 
