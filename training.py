import torch


def begin_training(model,num_epochs, train_loader, test_loader, loss, optimizer):

    training_loss = []
    validation_loss = []
    average_validation_loss = []
    for epoch in range(num_epochs):
        model.train(True)
        running_loss = 0
        print("Epoch: " + str(epoch))

        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0], batch[1]
            output = model(x_batch)
            loss_ = loss(output, y_batch)
            running_loss += loss_.item()

            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()


        print("Training Loss: " + str(running_loss))
        training_loss.append(running_loss)
        model.train(False)
        vad_loss = 0

        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0], batch[1]

            with torch.no_grad():
                output = model(x_batch)
                loss_ = loss(output, y_batch)
                vad_loss += loss_.item()

        validation_loss.append(vad_loss)
        avg_loss_across_batches = vad_loss / len(test_loader)
        average_validation_loss.append(avg_loss_across_batches)
        
        print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
        print('***************************************************')
        print('\n')

    return model, training_loss, validation_loss, average_validation_loss