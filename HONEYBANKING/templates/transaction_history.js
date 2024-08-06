function searchTransactions() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    const transactionType = document.getElementById('transaction-type').value;

    fetch('/search_transactions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ start_date: startDate, end_date: endDate, transaction_type: transactionType })
    })
    .then(response => response.json())
    .then(data => {
        const transactionList = document.getElementById('transaction-list');
        transactionList.innerHTML = '';

        if (data.length === 0) {
            transactionList.innerHTML = '<p>No transactions found.</p>';
        } else {
            data.forEach(transaction => {
                const transactionItem = document.createElement('div');
                transactionItem.className = 'transaction-item';
                transactionItem.innerHTML = `
                    <div class="details">
                        <p>Recipient: ${transaction.recipient_name} (${transaction.recipient_stk})</p>
                        <p>Description: ${transaction.description}</p>
                        <p>Timestamp: ${transaction.timestamp}</p>
                    </div>
                    <div class="amount ${transaction.amount < 0 ? 'negative' : 'positive'}">
                        ${transaction.amount} VND
                    </div>
                `;
                transactionList.appendChild(transactionItem);
            });
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
