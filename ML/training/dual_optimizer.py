import tensorflow as tf
from keras.saving import register_keras_serializable
import logging

logger = logging.getLogger("poker.dual_optimizer")

@register_keras_serializable(package="Poker")
class DualOptimizerModel(tf.keras.Model):
    """
    Wrapper that uses separate optimizers for encoder and head layers.
    
    Usage:
        base_model = build_model(config, mode_config)
        model = DualOptimizerModel(
            base_model,
            encoder_lr=1e-4,
            head_lr=1e-3
        )
        model.compile(loss=loss, loss_weights=loss_weights)
    """
    
    def __init__(self, base_model, encoder_lr=1e-4, head_lr=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self._encoder_lr = encoder_lr
        self._head_lr = head_lr
                
        # Create two optimizers
        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=encoder_lr)
        self.head_optimizer = tf.keras.optimizers.Adam(learning_rate=head_lr)
        
        # Split variables by layer type
        self._split_variables()
    
    def get_config(self):
        """Serialization config for saving/loading."""
        config = super().get_config()
        config.update({
            "encoder_lr": self._encoder_lr,
            "head_lr": self._head_lr,
            # Include the serialized base_model so Keras can reconstruct nested model and load weights
            "base_model_config": tf.keras.models.serialize(self.base_model),
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Construct an instance from config during Keras deserialization.

        When a DualOptimizerModel was saved directly, Keras will only supply
        the wrapper's config (encoder_lr/head_lr) but not the original
        `base_model` argument. Provide a safe placeholder base_model so the
        object can be deserialized and then allow the loader to unwrap the
        real base model/weights (if present) or for the caller to handle it.
        """
        encoder_lr = config.pop("encoder_lr", 1e-4)
        head_lr = config.pop("head_lr", 1e-3)

        # Try to reconstruct base_model if config contains a serialized model
        base_model = None
        base_cfg = config.pop("base_model_config", None)
        if base_cfg is not None:
            try:
                base_model = tf.keras.models.model_from_config(base_cfg)
            except Exception as e:
                logger.warning("Failed to reconstruct base_model from saved config: %s", e)
        else:
            # Warn callers that missing base_model_config will likely cause weight restoration failures
            logger.warning(
                "Deserializing DualOptimizerModel without embedded base_model_config. "
                "If this artifact was saved as a wrapper, weight restoration may fail; "
                "prefer saving the underlying base_model (model.base_model.save(...))."
            )

        # Fallback to a minimal placeholder base model if reconstruction failed
        if base_model is None:
            inp = tf.keras.Input(shape=(14, 4, 2))
            out = tf.keras.layers.Lambda(lambda x: x)(inp)
            base_model = tf.keras.Model(inputs=inp, outputs=out)

        return cls(base_model, encoder_lr=encoder_lr, head_lr=head_lr, **config)
    
    @property
    def optimizer(self):
        # Return a dummy optimizer, or choose one of the two optimizers
        # For safety, you can create a "no-op" optimizer or return one optimizer
        return self.encoder_optimizer  # or self.encoder_optimizer

    @optimizer.setter
    def optimizer(self, value):
        # Accept assignment but do nothing or handle if needed
        # This prevents the "no setter" error
        pass

    def _split_variables(self):
        """Split trainable variables into encoder and head groups."""
        from ML.models import CardStateEncoder, CardSetEncoder
        
        self.encoder_vars = []
        self.head_vars = []
        
        for layer in self.base_model.layers:
            # Check if layer is encoder type
            if isinstance(layer, (CardStateEncoder, CardSetEncoder)):
                self.encoder_vars.extend(layer.trainable_variables)
            else:
                # Check nested layers (for heads containing encoders)
                if hasattr(layer, 'layers'):
                    for sublayer in layer.layers:
                        if isinstance(sublayer, (CardStateEncoder, CardSetEncoder)):
                            self.encoder_vars.extend(sublayer.trainable_variables)
                        else:
                            self.head_vars.extend(sublayer.trainable_variables)
                else:
                    self.head_vars.extend(layer.trainable_variables)
        
        print(f"\n🔧 Variable Split:")
        print(f"  Encoder variables: {len(self.encoder_vars)}")
        print(f"  Head variables: {len(self.head_vars)}")
    
    def call(self, inputs, training=False):
        """Forward pass through base model."""
        return self.base_model(inputs, training=training)
    
    def train_step(self, data):
        """Custom training step with dual optimizers."""
        x, y = data
        
        # Use persistent=True to compute gradients multiple times
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            y_pred = self(x, training=True)
            
            # Compute loss
            loss = self.compute_loss(
                x=x,
                y=y,
                y_pred=y_pred,
                training=True,
            )
                
        # Compute gradients for encoder and head separately
        if self.encoder_vars:
            encoder_grads = tape.gradient(loss, self.encoder_vars)
            self.encoder_optimizer.apply_gradients(zip(encoder_grads, self.encoder_vars))
        
        if self.head_vars:
            head_grads = tape.gradient(loss, self.head_vars)
            self.head_optimizer.apply_gradients(zip(head_grads, self.head_vars))
        
        del tape

        # Update metrics
        for metric in self.metrics:
            try:
                metric.update_state(y, y_pred)
            except Exception:
                pass

        results = {}
        for metric in self.metrics:
            if hasattr(metric, "result"):
                try:
                    results[metric.name] = metric.result()
                except Exception:
                    continue

        results["loss"] = loss

        opt = self.encoder_optimizer
        if opt is not None and hasattr(opt, "learning_rate"):
            results["encoder_lr"] = opt.learning_rate

        if self.head_optimizer is not None:
            results["head_lr"] = self.head_optimizer.learning_rate
        
        return results
    
    def test_step(self, data):
        """Validation step (no optimization)."""
        x, y = data
        y_pred = self(x, training=False)

        loss = self.compute_loss(
            x=x,
            y=y,
            y_pred=y_pred,
            training=False,
        )

        for metric in self.metrics:
            try:
                metric.update_state(y, y_pred)
            except Exception:
                pass

        results = {}
        for metric in self.metrics:
            if hasattr(metric, "result"):
                try:
                    results[metric.name] = metric.result()
                except Exception:
                    continue

        results["loss"] = loss

        return results