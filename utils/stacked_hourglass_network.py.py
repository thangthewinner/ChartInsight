from tensorflow.keras import layers 
from tensorflow.keras.models import Model

def BottleNeck(inputs, filters, downsample = False):
    """ 
    Create a Bottleneck block for the Hourglass Network.
    
    Args:
        inputs (tensor): Input tensor.
        filters (int): Number of filters (channels) in the Conv2D layer.
        downsample (bool, default: False): If True, downsamples the input tensor to match the output tensor for the skip connection

    Returns:
        tensor: Output tensor after the Bottleneck block.
    """ 
    if downsample:         # If downsample is True, reduce the size of the input tensor to match the output tensor.
        inputs = layers.Conv2D(filters = filters, kernel_size = 1, padding = "same")(inputs)
    
    x = layers.Conv2D(filters=filters//2, kernel_size=1, activation="relu")(inputs)
    x = layers.Conv2D(filters=filters//2, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters=filters, kernel_size=1, activation="relu")(x)

    # Skip connection between the input and output tensors of the block.
    return layers.Add()([x, inputs])
    
def HourglassModule(inputs, order, filters, num_residual = 3):
    """ 
    Create an Hourglass Module for the Hourglass Network.

    Args:
        inputs (tensor): Input tensor.
        order (int): The order (depth) of the Hourglass module.
        filters (int): Number of filters (channels) in the Conv2D layers.
        num_residual (int): Number of Bottleneck blocks in each branch of the Hourglass. Default is 3.

    Returns:
        tensor: Output tensor of the Hourglass module.
    """ 
    # Upper branch of the Hourglass: starts with a Bottleneck followed by num_residual Bottleneck blocks.
    up1 = BottleNeck(inputs, filters, downsample = True)
    for _ in range(num_residual):
        up1 = BottleNeck(up1, filters)
        
    # Lower branch of the Hourglass: starts with MaxPooling followed by Bottleneck.
    low1 = layers.MaxPool2D(pool_size=2, strides=2)(inputs)
    low1 = BottleNeck(low1, filters, downsample = True)
    for _ in range(num_residual - 1):
        low1 = BottleNeck(low1, filters)
        
    # Deeper Hourglass layers (if order > 1), recursively call HourglassModule.
    low2 = low1 
    if order > 1:
        low2 = HourglassModule(low1, order - 1, filters)
    else:
        # Continue with num_residual Bottleneck blocks.
        for _ in range(num_residual):
            low2 = BottleNeck(low2, filters)
         
    # Bottom branch: continues with num_residual Bottleneck blocks.
    low3 = low2
    for _ in range(num_residual):
        low3 = BottleNeck(low3, filters)
    
    # Upsampling the lower branch to match the size of the upper branch.
    up2 = layers.UpSampling2D(size=2)(low3)
    
    # Add the upper and lower branches together.
    return layers.Add()([up1, up2])

def LinearLayer(inputs, filters):
    """ 
    Create a Linear layer for the Hourglass Network.

    Args:
        inputs (tensor): Input tensor.
        filters (int): Number of filters (channels) in the Conv2D layer.

    Returns:
        tensor: Output tensor after passing through the Linear layer.
    """ 
    x = layers.Conv2D(filters = filters,
                      kernel_size = 1,
                      padding = "same",
                      kernel_initializer = "he_normal")(inputs)
    x = layers.BatchNormalization(momentum = 0.9)(x)
    x = layers.ReLU()(x)
    
    return x

def StackedHourglassNetwork(input_shape, num_stack, num_residual, num_heatmap):
    """ 
    Build a Stacked Hourglass Network.

    Args:
        input_shape (tuple): Shape of the input image.
        num_stack (int): Number of Hourglass stacks.
        num_residual (int): Number of Bottleneck blocks in each Hourglass module.
        num_heatmap (int): Number of output heatmaps (channels).

    Returns:
        model: Stacked Hourglass model.
    """
    inputs = layers.Input(shape = input_shape)
    
    # Initial processing
    x = layers.Conv2D(filters = 64, 
                      kernel_size = 7,
                      strides = 2, 
                      padding = "same",
                      kernel_initializer = "he_normal")(inputs)
    x = layers.BatchNormalization(momentum = 0.9)(x)
    x = layers.Activation("relu")(x)
    x = BottleNeck(x, 128, downsample = True)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = BottleNeck(x, 128)
    x = BottleNeck(x, 256, downsample = True)
    
    # Stacking Hourglass blocks
    ys = []
    for i in range(num_stack):
        # Pass through HourglassModule
        x = HourglassModule(x, order = 4, filters = 256, num_residual = num_residual)
        for j in range(num_residual):
            x = BottleNeck(x, 256)
        
        # Predict output (heatmap) after each stack
        x = LinearLayer(x, 256)
        y = layers.Conv2D(filters = num_heatmap,
                          kernel_size = 1,
                          padding = "same",
                          kernel_initializer = "he_normal")(x)
        ys.append(y)
        
        # If it's not the last stack, add predictions back to the input for the next stack.
        if i < num_stack - 1 :
            y_intermediate_1 = layers.Conv2D(filters = 256, kernel_size = 1, strides =1 )(x)
            y_intermediate_2 = layers.Conv2D(filters = 256, kernel_size = 1, strides =1 )(y)
            x = layers.Add()([x, y_intermediate_1, y_intermediate_2])
        
    # Return model with inputs and the outputs as the heatmaps.
    return Model(inputs = inputs, outputs = ys, name = "StackedHourglass")